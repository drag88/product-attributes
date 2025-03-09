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

from .api_service import APIService
from src.base.utils import (
    create_image_message,
    image_to_data_url
)
from src.utils.validation import (
    validate_enum_field,
    validate_enum_list
)


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
        
        # Validate Cohere config
        if not self.cohere_config.get("model"):
            logger.critical("Missing Cohere model configuration")
            raise ValueError("Cohere model configuration required")
        
        if "embed" not in self.cohere_config.get("supported_operations", []):
            logger.warning("Cohere config may not support embed operations")
        
        logger.info(
            f"AttributeGenerator initialized with model: {self.api_service.model}"
        )

    def register_enum_mapping(
        self, 
        field_name: str, 
        enum_class: Type[Enum]
    ) -> None:
        """Register an enum class for a field."""
        self._enum_mappings[field_name] = enum_class

    def _load_prompt(self, product_type: str) -> str:
        # Convert to lowercase and append _prompt suffix
        prompt_filename = f"{product_type.lower()}_prompt"
        prompt_path = self.prompts_dir / f"{prompt_filename}.txt"
        try:
            with open(prompt_path) as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {prompt_path}")
            return ""

    def _format_product_details(self, product_data: Dict[str, Any]) -> str:
        return json.dumps(product_data, indent=2)

    def _process_attributes(
        self, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and validate attributes using registered enum mappings."""
        result = {}
        
        for field, value in data.items():
            if field in self._enum_mappings:
                enum_class = self._enum_mappings[field]
                if isinstance(value, list):
                    result[field] = validate_enum_list(
                        value, 
                        enum_class, 
                        field
                    )
                else:
                    result[field] = validate_enum_field(
                        value, 
                        enum_class, 
                        field
                    )
            else:
                result[field] = value
                
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
            
            # Process attributes
            print(f"Processing attributes for product {product_id}")
            logger.debug(f"Processing attributes for product {product_id}")
            try:
                json_data = self._extract_json_response(content)
                attributes = self._process_attributes(json_data)
                print(f"Successfully processed attributes for product {product_id}")
                logger.debug(f"Successfully processed attributes for product {product_id}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response for product {product_id}: {str(e)}")
                logger.error(f"Invalid JSON response: {str(e)}")
                return None
            
            # Add metadata
            print(f"Adding metadata for product {product_id}")
            logger.debug(f"Adding metadata for product {product_id}")
            attributes["inference_time"] = time.time() - start_time
            attributes["model"] = self.api_service.model
            
            # Add token usage if available
            if hasattr(response, 'usage'):
                print(f"Adding token usage for product {product_id}")
                logger.debug(f"Adding token usage for product {product_id}")
                attributes["input_tokens"] = response.usage.input_tokens
                attributes["output_tokens"] = response.usage.output_tokens
            
            # Add embeddings to attributes (always include keys)
            attributes.update({
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "has_text_embedding": text_embedding is not None,
                "has_image_embedding": image_embedding is not None
            })
            
            # Add after getting attributes
            required_fields = ['title', 'description', 'primary_color', 'material']
            if not all(field in attributes for field in required_fields):
                missing = [f for f in required_fields if f not in attributes]
                logger.error(f"Missing required fields {missing} in response for {product_id}")
                return None
            
            # Move image validation BEFORE returning attributes
            if image_path:
                logger.debug(f"Image path validation: {Path(image_path).exists()}")
                if not Path(image_path).exists():
                    logger.error(f"Missing image file: {image_path}")
                    return None  # Fail early if image missing
            
            # Add validation before storing embeddings
            if image_embedding:
                attributes["image_embedding"] = image_embedding
                logger.debug(f"Image embedding length: {len(image_embedding)}")
            else:
                logger.warning("No image embedding generated")
            
            # This should be the LAST operation before return
            print(f"Attribute generation completed for product {product_id}")
            logger.debug(f"Attribute generation completed for product {product_id}")
            return attributes
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error for product {product_id}: {str(e)}")
            logger.error(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error generating attributes for product {product_id}: {str(e)}")
            logger.error(f"Error generating attributes: {str(e)}", exc_info=True)
            return None 

    def _extract_json_response(self, content: str) -> dict:
        """Extract JSON content from response wrapped in <json> tags"""
        try:
            json_match = re.search(r'<json>(.*?)</json>', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            # Fallback to entire content if no tags found
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed. Content: {content[:200]}...")
            raise 