from typing import Dict, Any, Type, List, Optional, Tuple
import json
import logging
from pathlib import Path
import re
from enum import Enum
from functools import lru_cache
import base64

from .api_service import APIService
from src.base.utils import (
    validate_enum_value,
    validate_enum_list,
    create_image_message
)


logger = logging.getLogger(__name__)


class AttributeGenerator:
    def __init__(self, api_service: APIService, config: Dict):
        self.api_service = api_service
        self.config = config
        self.prompts_dir = Path(config["prompts"]["dir"])
        self._enum_mappings = {}
        
        # Extract API configs
        self.anthropic_config = config["api"]["anthropic"]
        self.cohere_config = config["api"]["cohere"]

    def register_enum_mapping(
        self, 
        field_name: str, 
        enum_class: Type[Enum]
    ) -> None:
        """Register an enum class for a field."""
        self._enum_mappings[field_name] = enum_class

    def _load_product_prompt(self, product_type: str) -> str:
        prompt_file = self.prompts_dir / f"{product_type}_prompt.txt"
        if not prompt_file.exists():
            raise ValueError(f"No prompt template found for {product_type}")
        return prompt_file.read_text()

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
                    result[field] = validate_enum_value(
                        value, 
                        enum_class, 
                        field
                    )
            else:
                result[field] = value
                
        return result

    async def _generate_image_embedding(
        self,
        image_path: str
    ) -> Optional[List[float]]:
        """Generate image embedding using Cohere API."""
        try:
            data_url = self._image_to_data_url(image_path)
            response = await self.api_service.call_cohere_api(
                method='embed',
                model=self.cohere_config["model"],
                texts=[],
                images=[data_url],
                input_type="image",
                embedding_types=["float"]
            )
            return response.embeddings.float[0]
        except Exception as e:
            logger.error(
                f"Failed to generate image embedding: {str(e)}", 
                exc_info=True
            )
            return None

    def _image_to_data_url(self, image_path: str) -> str:
        """Convert image to data URL format for Cohere API."""
        with open(image_path, "rb") as f:
            data = f.read()
            file_type = Path(image_path).suffix[1:]
            base64_str = base64.b64encode(data).decode("utf-8")
            return f"data:image/{file_type};base64,{base64_str}"

    async def generate_attributes(
        self,
        product_data: Dict[str, Any],
        product_type: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            SYSTEM_PROMPT = self._load_product_prompt(product_type)
            product_details = self._format_product_details(product_data)

            # Generate image embedding if image provided
            image_embedding = None
            if image_path:
                image_embedding = await self._generate_image_embedding(image_path)
                if not image_embedding:
                    logger.warning("Failed to generate image embedding")

            # Prepare message content
            content = []
            if image_path:
                content.append(create_image_message(image_path))
            content.append({
                "type": "text",
                "text": f"Product Details:\n{product_details}"
            })

            messages = [{
                "role": "user",
                "content": content
            }]

            # Call Anthropic API with config-driven parameters
            response = await self.api_service.call_api(
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=messages,
                model=self.anthropic_config["model"],
                temperature=0,
                max_tokens=self.config["prompts"]["max_tokens"],
                extra_headers={
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "anthropic-metadata": json.dumps({
                        "task_type": "attribute_extraction",
                        "requires_consistency": True
                    })
                }
            )

            try:
                content = response.content[0].text if response.content else ""
                json_match = re.search(
                    r'<json>(.*?)</json>', 
                    content, 
                    re.DOTALL
                )
                if not json_match:
                    raise ValueError("No JSON found in response")

                raw_attributes = json.loads(json_match.group(1))
                attributes = self._process_attributes(raw_attributes)
                
                # Add image embedding to attributes if available
                if image_embedding is not None:
                    attributes['image_embedding'] = image_embedding
                
                cost_info = self.api_service.calculate_cost(response)
                logger.info(f"Generation cost: {cost_info}")
                return attributes
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse API response: {e}")
                raise

        except Exception as e:
            logger.error(f"Error generating attributes: {e}")
            raise 