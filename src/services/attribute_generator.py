from typing import Dict, Any, Type, List, Optional, Tuple
import json
import logging
from pathlib import Path
import re
from enum import Enum
from functools import lru_cache
import base64
import time
import pandas as pd
import traceback
from importlib import import_module
from src.base.clothing_item import ClothingItem

from .api_service import APIService
from src.base.utils import (
    create_image_message,
    image_to_data_url
)
from src.utils.validation import (
    validate_enum_field,
    validate_enum_list
)
from src.utils.config_loader import ConfigManager


logger = logging.getLogger(__name__)

# Temporarily add for debugging
logger.setLevel(logging.DEBUG)
logging.getLogger("src.services.api_service").setLevel(logging.DEBUG)

# Force debug logging for this module
logger.setLevel(logging.DEBUG)
logging.getLogger("src.services.attribute_generator").setLevel(logging.DEBUG)
logging.getLogger("src.services.api_service").setLevel(logging.DEBUG)


class AttributeGenerator:
    def __init__(self, api_service: APIService, config: Dict):
        self.api_service = api_service
        self.config = config
        # Resolve prompts directory relative to project root
        self.prompts_dir = Path(__file__).parent.parent.parent / config.get("prompts", {}).get("dir", "config/prompts")
        self._enum_mappings = {}
        
        # Extract API configs with defaults
        api_config = config.get("api", {})
        self.anthropic_config = api_config.get("anthropic", {})
        self.cohere_config = api_config.get("cohere", {})
        
        # Set default values if not provided
        self.max_tokens = self.anthropic_config.get(
            "max_tokens", 
            config.get("prompts", {}).get("max_tokens", 4096)
        )
        self.temperature = self.anthropic_config.get(
            "temperature", 
            config.get("prompts", {}).get("system_temperature", 0.1)
        )
        
        # Only validate Cohere config if api_service is provided
        if api_service is not None:
            if not self.cohere_config.get("model"):
                logger.critical("Missing Cohere model configuration")
                raise ValueError("Cohere model configuration required")
            
            if "embed" not in self.cohere_config.get("supported_operations", []):
                logger.warning("Cohere config may not support embed operations")
        
        # Register enums from configuration
        self._register_enums_from_config()
        
        # Only log model info if api_service is provided
        if api_service is not None:
            logger.info(
                f"AttributeGenerator initialized with model: {self.api_service.model}"
            )
        else:
            logger.info("AttributeGenerator initialized in test mode without API service")

    def _register_enums_from_config(self) -> None:
        """Register enums for all product types based on configuration."""
        try:
            # Get base attributes that apply to all products
            base_attributes = self.config.get('attributes', {})
            self._register_attributes_enums(base_attributes, 'base')
            
            # Load and register product-specific enums
            config_dir = Path(__file__).parent.parent.parent / 'config'
            for config_file in config_dir.glob('*_config.yaml'):
                if config_file.stem == 'base_config':
                    continue
                    
                product_type = config_file.stem.replace('_config', '')
                product_config = ConfigManager.get_product_config(product_type)
                
                if product_config and 'attributes' in product_config:
                    logger.debug(f"Registering enums for {product_type}")
                    self._register_attributes_enums(
                        product_config['attributes'],
                        product_type
                    )
                    
        except Exception as e:
            logger.error(f"Error registering enums from config: {str(e)}")
            raise

    def _register_attributes_enums(
        self,
        attributes: Dict[str, Dict],
        source: str
    ) -> None:
        """Register enums for a set of attributes."""
        for field_name, field_config in attributes.items():
            if not isinstance(field_config, dict):
                continue
                
            # Check if this is an enum field
            if field_config.get('type') == 'enum':
                self._register_enum_for_field(
                    field_name,
                    field_config.get('values'),
                    source
                )
            
            # Check for list fields with enum items
            elif (field_config.get('type') == 'list' and 
                  field_config.get('item_type') == 'enum'):
                self._register_enum_for_field(
                    field_name,
                    field_config.get('values'),
                    source
                )

    def _register_enum_for_field(
        self, 
        field_name: str, 
        enum_name: str,
        source: str
    ) -> None:
        """Register a single enum for a field."""
        try:
            # Try to import from base.enums first
            try:
                enum_module = import_module('src.base.enums')
                enum_class = getattr(enum_module, enum_name)
            except (ImportError, AttributeError):
                # If not in base.enums, try product-specific enums
                try:
                    enum_module = import_module(f'src.models.{source}.enums')
                    enum_class = getattr(enum_module, enum_name)
                except (ImportError, AttributeError):
                    logger.error(
                        f"Could not find enum class {enum_name} "
                        f"for field {field_name} in {source}"
                    )
                    return
            
            if issubclass(enum_class, Enum):
                self._enum_mappings[field_name] = enum_class
                logger.debug(
                    f"Registered enum {enum_name} for field {field_name} "
                    f"from {source}"
                )
            else:
                logger.error(
                    f"{enum_name} is not an Enum class "
                    f"for field {field_name} in {source}"
                )
                
        except Exception as e:
            logger.error(
                f"Error registering enum {enum_name} for field {field_name}: {str(e)}"
            )

    def _load_prompt(self, product_type: str) -> str:
        """Load and populate prompt with dynamic enum values."""
        try:
            # Convert to lowercase and append _prompt suffix
            prompt_filename = f"{product_type.lower()}_prompt"
            prompt_path = self.prompts_dir / f"{prompt_filename}.txt"
            
            with open(prompt_path) as f:
                prompt = f.read().strip()

            # Get all enums from src.base.enums
            enum_module = import_module('src.base.enums')
            
            # Replace enum placeholders in the prompt
            for name, obj in vars(enum_module).items():
                if isinstance(obj, type) and issubclass(obj, Enum) and obj != Enum:
                    # Create formatted string of enum values
                    enum_values = "'" + "', '".join(v.value for v in obj) + "'"
                    # Replace both patterns
                    prompt = prompt.replace(f"{{', '.join({name}.__members__.keys())}}", enum_values)
                    prompt = prompt.replace(f"{{{name}}}", enum_values)

            return prompt
            
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt: {str(e)}\n{traceback.format_exc()}")
            return ""

    def _format_product_details(self, product_data: Dict[str, Any]) -> str:
        return json.dumps(product_data, indent=2)

    def _process_attributes(
        self, 
        data: Dict[str, Any],
        product_type: str
    ) -> Dict[str, Any]:
        """Process and validate attributes from raw data.
        
        Args:
            data: Raw attribute data from LLM
            product_type: Type of product (e.g., 'saree', 'blouse')
            
        Returns:
            Dict of validated attributes
        """
        result = {}
        
        # Get product config and its attributes
        product_config = ConfigManager.get_product_config(product_type)
        defined_attributes = product_config.get('attributes', {})
        required_attrs = ConfigManager.get_required_attributes(product_type)
        
        # First, add all fields from the data to the result
        # This ensures we don't lose any data generated by the LLM
        for field, value in data.items():
            result[field] = value
        
        # Then process and validate fields defined in the config
        for field, value in data.items():
            # Skip fields that aren't defined in the product config but keep them in result
            if field not in defined_attributes:
                logger.debug(f"Skipping undefined field '{field}' for {product_type}")
                continue
                
            if field in self._enum_mappings:
                enum_class = self._enum_mappings[field]
                # Get threshold from attribute config, default to 0.8 if not specified
                threshold = defined_attributes[field].get('threshold', 0.8)
                
                try:
                    if isinstance(value, list):
                        result[field] = validate_enum_list(
                            value, 
                            enum_class, 
                            field,
                            threshold=threshold
                        )
                    else:
                        result[field] = validate_enum_field(
                            value, 
                            enum_class, 
                            field,
                            threshold=threshold
                        )
                except Exception as e:
                    logger.error(f"Error validating field {field}: {str(e)}")
                    if field in required_attrs:
                        logger.error(f"Failed to validate required field {field}")
                        # Don't return None, keep processing other fields
                        # Just log the error and continue
            
        # Check for missing required attributes and try to set defaults
        for field in required_attrs:
            if field not in result or result[field] is None or (isinstance(result[field], list) and not result[field]):
                # Try to get default from config
                if field in defined_attributes and 'default' in defined_attributes[field]:
                    result[field] = defined_attributes[field]['default']
                    logger.debug(f"Set default value for required field {field}")
                else:
                    logger.warning(f"Missing required field {field} with no default")
        
        return result

    async def _generate_image_embedding(self, image_path: str) -> Optional[List[float]]:
        try:
            data_url = image_to_data_url(image_path)
            if not data_url:
                logger.error(f"Failed to convert image to data URL: {image_path}")
                return None
            
            logger.debug(f"Sending image to Cohere API: {image_path}")
            
            # CRITICAL FIX: Change input_type to "image" for image embeddings
            response = await self.api_service.call_cohere_api(
                method="embed",
                images=[data_url],
                model=self.cohere_config.get("model"),
                input_type="image",  # Changed from "search_document" to "image"
                embedding_types=["float"]
            )
            
            # Add extensive logging to understand response structure
            logger.debug(f"Image embedding response type: {type(response)}")
            
            # Dump full response structure for debugging
            if hasattr(response, '__dict__'):
                logger.debug(f"Response attributes: {dir(response)}")
                if hasattr(response, 'embeddings'):
                    logger.debug(f"Embeddings attributes: {dir(response.embeddings)}")
                
            # Try different ways to access embeddings based on Cohere docs
            if hasattr(response, 'embeddings'):
                embeddings = response.embeddings
                
                # Try float_ (with underscore) as shown in some Cohere examples
                if hasattr(embeddings, 'float_') and isinstance(embeddings.float_, list) and len(embeddings.float_) > 0:
                    logger.debug(f"Found float_ embeddings of length {len(embeddings.float_)}")
                    return embeddings.float_[0]
                
                # Try float (without underscore)
                if hasattr(embeddings, 'float') and isinstance(embeddings.float, list) and len(embeddings.float) > 0:
                    logger.debug(f"Found float embeddings of length {len(embeddings.float)}")
                    return embeddings.float[0]
                
                # Check for list structure
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    logger.debug(f"Found list-type embeddings of length {len(embeddings)}")
                    return embeddings[0]
                
                # Try to access as a dictionary
                if isinstance(embeddings, dict) and 'float' in embeddings and len(embeddings['float']) > 0:
                    logger.debug(f"Found dictionary float embeddings of length {len(embeddings['float'])}")
                    return embeddings['float'][0]
                
                if isinstance(embeddings, dict) and 'float_' in embeddings and len(embeddings['float_']) > 0:
                    logger.debug(f"Found dictionary float_ embeddings of length {len(embeddings['float_'])}")
                    return embeddings['float_'][0]
            
            # Direct attribute access in case of different structure
            if hasattr(response, 'float') and isinstance(response.float, list) and len(response.float) > 0:
                logger.debug(f"Found top-level float embeddings of length {len(response.float)}")
                return response.float[0]
            
            # Last resort - try to parse as dictionary
            if isinstance(response, dict):
                if 'embeddings' in response:
                    emb = response['embeddings']
                    if isinstance(emb, dict):
                        if 'float_' in emb and isinstance(emb['float_'], list) and len(emb['float_']) > 0:
                            logger.debug(f"Found dictionary path embeddings['float_'] of length {len(emb['float_'])}")
                            return emb['float_'][0]
                        if 'float' in emb and isinstance(emb['float'], list) and len(emb['float']) > 0:
                            logger.debug(f"Found dictionary path embeddings['float'] of length {len(emb['float'])}")
                            return emb['float'][0]
            
            # Log detailed response for debugging
            logger.error(f"Could not extract embeddings from response: {response}")
            return None
        
        except Exception as e:
            logger.exception(f"Error generating image embedding: {str(e)}")
            return None

    async def _generate_text_embedding(
        self,
        text: str
    ) -> Optional[List[float]]:
        try:
            # Call Cohere API for text embedding
            logger.debug(f"Generating text embedding for text of length: {len(text)}")
            
            # Use correct input_type for text
            response = await self.api_service.call_cohere_api(
                method="embed",
                texts=[text],
                model=self.cohere_config.get("model"),
                input_type="search_document",  # This is correct for text
                embedding_types=["float"]
            )
            
            # Add extensive logging to understand response structure
            logger.debug(f"Text embedding response type: {type(response)}")
            
            # Dump full response structure for debugging
            if hasattr(response, '__dict__'):
                logger.debug(f"Response attributes: {dir(response)}")
                if hasattr(response, 'embeddings'):
                    logger.debug(f"Embeddings attributes: {dir(response.embeddings)}")
            
            # Try different ways to access embeddings based on Cohere docs
            if hasattr(response, 'embeddings'):
                embeddings = response.embeddings
                
                # Try float_ (with underscore) as shown in some Cohere examples
                if hasattr(embeddings, 'float_') and isinstance(embeddings.float_, list) and len(embeddings.float_) > 0:
                    logger.debug(f"Found float_ embeddings of length {len(embeddings.float_)}")
                    return embeddings.float_[0]
                
                # Try float (without underscore)
                if hasattr(embeddings, 'float') and isinstance(embeddings.float, list) and len(embeddings.float) > 0:
                    logger.debug(f"Found float embeddings of length {len(embeddings.float)}")
                    return embeddings.float[0]
                
                # Check for list structure
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    logger.debug(f"Found list-type embeddings of length {len(embeddings)}")
                    return embeddings[0]
                
                # Try to access as a dictionary
                if isinstance(embeddings, dict) and 'float' in embeddings and len(embeddings['float']) > 0:
                    logger.debug(f"Found dictionary float embeddings of length {len(embeddings['float'])}")
                    return embeddings['float'][0]
                
                if isinstance(embeddings, dict) and 'float_' in embeddings and len(embeddings['float_']) > 0:
                    logger.debug(f"Found dictionary float_ embeddings of length {len(embeddings['float_'])}")
                    return embeddings['float_'][0]
            
            # Direct attribute access in case of different structure
            if hasattr(response, 'float') and isinstance(response.float, list) and len(response.float) > 0:
                logger.debug(f"Found top-level float embeddings of length {len(response.float)}")
                return response.float[0]
            
            # Last resort - try to parse as dictionary
            if isinstance(response, dict):
                if 'embeddings' in response:
                    emb = response['embeddings']
                    if isinstance(emb, dict):
                        if 'float_' in emb and isinstance(emb['float_'], list) and len(emb['float_']) > 0:
                            logger.debug(f"Found dictionary path embeddings['float_'] of length {len(emb['float_'])}")
                            return emb['float_'][0]
                        if 'float' in emb and isinstance(emb['float'], list) and len(emb['float']) > 0:
                            logger.debug(f"Found dictionary path embeddings['float'] of length {len(emb['float'])}")
                            return emb['float'][0]
            
            # Log detailed response for debugging
            logger.error(f"Could not extract embeddings from response: {response}")
            return None
        except Exception as e:
            logger.exception(f"Error generating text embedding: {str(e)}")
            return None

    async def generate_attributes(
        self,
        product_data: Dict,
        product_type: str,
        image_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate attributes for a product using AI.
        
        Args:
            product_data: Product data dictionary
            product_type: Type of product (e.g., 'saree', 'blouse')
            image_path: Optional path to product image
            
        Returns:
            Dictionary of generated attributes
        """
        product_id = product_data.get('product_id', 'unknown')
        try:
            # Add validation at start
            if not isinstance(product_data.get('product_id'), (str, int)):
                logger.error(f"Invalid product ID: {product_data.get('product_id')}")
                return None
            
            if pd.isna(product_data.get('product_type')):
                logger.error("Missing product_type in product data")
                return None
            
            start_time = time.time()
            print(f"Starting attribute generation for product {product_id}")
            logger.info(f"Generating attributes for product {product_id}")
            
            # Validate image path early
            if image_path:
                logger.debug(f"Validating image path: {image_path}")
                if not Path(image_path).exists():
                    logger.error(f"Image file does not exist: {image_path}")
                    image_path = None  # Reset so we don't try to use it later
                elif not Path(image_path).is_file():
                    logger.error(f"Image path is not a file: {image_path}")
                    image_path = None
                else:
                    file_size_mb = Path(image_path).stat().st_size / (1024 * 1024)
                    logger.debug(f"Image file size: {file_size_mb:.2f}MB")
                    if file_size_mb > 10:
                        logger.warning(f"Image file is very large ({file_size_mb:.2f}MB)")
            
            # Load product-specific prompt
            print(f"Loading prompt for product type: {product_type}")
            logger.debug(f"Loading prompt for product type: {product_type}")
            prompt = self._load_prompt(product_type)
            logger.debug(f"Prompt: {prompt}")
            if not prompt:
                print(f"Failed to load prompt for product type: {product_type}")
                logger.error(f"Failed to load prompt for product type: {product_type}")
                return None
            
            # Format product details for prompt
            print(f"Formatting product details for product {product_id}")
            logger.debug(f"Formatting product details for product {product_id}")
            product_details = self._format_product_details(product_data)
            
            # Prepare messages for API call
            print(f"Preparing messages for API call for product {product_id}")
            logger.debug(f"Preparing messages for API call for product {product_id}")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": product_details}
            ]
            
            # Add image if provided
            if image_path:
                print(f"Processing image for product {product_id}: {image_path}")
                logger.debug(f"Processing image for product {product_id}: {image_path}")
                try:
                    image_data_url = image_to_data_url(image_path)
                    if image_data_url:
                        print(f"Successfully converted image to data URL for product {product_id}")
                        logger.debug(f"Successfully converted image to data URL for product {product_id}")
                        # Replace text message with image message
                        messages[1] = create_image_message(image_data_url)
                except Exception as e:
                    print(f"Error processing image for product {product_id}: {str(e)}")
                    logger.error(f"Error processing image: {str(e)}")
            
            # Generate embeddings
            text_embedding = await self._generate_text_embedding(product_details)
            image_embedding = None
            
            if image_path:
                logger.debug(f"Generating image embedding for product {product_id}")
                image_embedding = await self._generate_image_embedding(image_path)
                if image_embedding is None:
                    logger.warning(f"Failed to generate image embedding for {image_path}")
            
            # Call API
            print(f"Calling API for product {product_id} with model {self.api_service.model}")
            logger.info(f"Calling API for product {product_id}")
            try:
                response = await self.api_service.call_api(
                    model=self.api_service.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                print(f"API call completed for product {product_id}")
                logger.debug(f"API call completed for product {product_id}")
            except Exception as e:
                print(f"API call failed for product {product_id}: {str(e)}")
                logger.error(f"API call failed: {str(e)}")
                return None
            
            # Process response
            if not response or not hasattr(response, 'content') or not response.content:
                print(f"Empty response from API for product {product_id}")
                logger.error("Empty response from API")
                return None
            
            # Extract content
            print(f"Extracting content from response for product {product_id}")
            logger.debug(f"Extracting content from response for product {product_id}")
            content = response.content[0].text
            logger.debug(f"Content: {content}")
            
            # Process attributes
            print(f"Processing attributes for product {product_id}")
            logger.debug(f"Processing attributes for product {product_id}")
            try:
                json_data = self._extract_json_response(content)
                # Pass product_type to _process_attributes
                attributes = self._process_attributes(json_data, product_type)
                if not attributes:  # Check if we got any attributes
                    logger.error(f"No valid attributes generated for {product_id}")
                    return None
                    
                # Add metadata
                attributes["inference_time"] = time.time() - start_time
                attributes["model"] = self.api_service.model
                
                # Add token usage if available
                if hasattr(response, 'usage'):
                    attributes["input_tokens"] = response.usage.input_tokens
                    attributes["output_tokens"] = response.usage.output_tokens
                
                # Add embeddings
                attributes.update({
                    "brand": ConfigManager.get_product_config(product_type)['attributes']['brand']['default'],
                    "brand_title": product_data.get("title"),
                    "price": product_data.get("price"),
                    "size": product_data.get("size"),
                    "product_image_link": product_data.get("product_image_link"),
                    "product_url": product_data.get("product_url"),
                    "text_embedding": text_embedding,
                    "image_embedding": image_embedding,
                    "has_text_embedding": text_embedding is not None,
                    "has_image_embedding": image_embedding is not None,
                    "product_type": product_type,
                    # "search_context" : ClothingItem.build_search_context(attributes)
                })
                logger.debug(f"Search context: {attributes['search_context']}") 

                # Log what fields were generated vs required (for debugging)
                required_attrs = ConfigManager.get_required_attributes(product_type)
                generated_fields = set(attributes.keys())
                missing_fields = [attr for attr in required_attrs if attr not in generated_fields]
                empty_fields = [attr for attr in required_attrs if attr in generated_fields and not attributes[attr]]
                
                if missing_fields or empty_fields:
                    logger.debug(
                        f"Product {product_id} attribute status:\n"
                        f"Generated fields: {sorted(generated_fields)}\n"
                        f"Required fields: {sorted(required_attrs)}\n"
                        f"Missing fields: {sorted(missing_fields)}\n"
                        f"Empty fields: {sorted(empty_fields)}"
                    )
                
                # Return the attributes without validation - validation will happen at save time
                print(f"Successfully processed attributes for product {product_id}")
                logger.debug(f"Successfully processed attributes for product {product_id}")
                return attributes
                
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response for product {product_id}: {str(e)}")
                logger.error(f"Invalid JSON response: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Error generating attributes for product {product_id}: {str(e)}")
            logger.error(f"Error generating attributes: {str(e)}", exc_info=True)
            return None

    def _extract_json_response(self, content: str) -> dict:
        """Extract JSON from LLM response.
        
        Args:
            content: Raw response content from LLM
            
        Returns:
            Extracted JSON as dict or empty dict if extraction failed
        """
        try:
            # Try to find JSON within <json> tags first
            json_pattern = r'<json>(.*?)</json>'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try to find JSON within ```json blocks
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try to find any JSON block with curly braces
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0).strip()
                return json.loads(json_str)
            
            # If no JSON found, try to parse the entire content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.error(f"Content: {content}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {} 