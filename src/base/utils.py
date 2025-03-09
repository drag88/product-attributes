from enum import Enum
from typing import Type, Dict, List, Any, Set, Optional
import base64
import mimetypes
import logging
import yaml
from pathlib import Path
import importlib
from functools import lru_cache
from src.base.enums import (
    Color, ColorDetailed, Pattern, Material, 
    EmbellishmentLevel, Embellishment, Occasion, 
    Style, Gender, AgeGroup
)
import os

logger = logging.getLogger(__name__)


class EnumRegistry:
    _instance = None
    _enum_mappings: Dict[str, Dict[str, Type[Enum]]] = {}
    _base_mappings: Dict[str, Type[Enum]] = {}
    _product_types: Set[str] = set()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Don't initialize immediately to avoid circular imports
            cls._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        
        # Initialize if not already done
        if not cls._initialized:
            cls._load_mappings_from_config()
            cls._initialized = True
            
        return cls._instance

    @classmethod
    def _load_mappings_from_config(cls) -> None:
        """Load enum mappings from configuration files."""
        try:
            # Import here to avoid circular imports
            from src.utils.config_loader import ProductConfigManager
            config_manager = ProductConfigManager.get_instance()
            
            # Load base mappings
            base_config = config_manager.get_base_config()
            cls._base_mappings = cls._extract_enum_mappings_from_attributes(
                base_config.get('attributes', {})
            )
            
            # Load product-specific mappings
            for product_type in config_manager.get_available_product_types():
                cls._product_types.add(product_type)
                
                product_config = config_manager.get_product_config(product_type)
                product_mappings = cls._extract_enum_mappings_from_attributes(
                    product_config.get('attributes', {})
                )
                
                if product_mappings:
                    cls.register_product_enums(product_type, product_mappings)
                    
        except Exception as e:
            logger.error(f"Error loading enum mappings from config files: {e}")
    
    @classmethod
    def _extract_enum_mappings_from_attributes(
        cls, attributes: Dict[str, Any]
    ) -> Dict[str, Type[Enum]]:
        """Extract enum mappings from attribute definitions."""
        enum_mappings = {}
        
        # Get enum module dynamically
        enums_module = importlib.import_module('src.base.enums')
        
        for field_name, attr_config in attributes.items():
            # Check if this is an enum field
            if attr_config.get('type') == 'enum':
                enum_class_name = attr_config.get('values')
                if enum_class_name:
                    try:
                        enum_class = getattr(enums_module, enum_class_name)
                        if issubclass(enum_class, Enum):
                            enum_mappings[field_name] = enum_class
                    except (AttributeError, TypeError):
                        logger.warning(
                            f"Could not find enum class '{enum_class_name}' "
                            f"for field '{field_name}'"
                        )
            
            # Check for list fields with enum items
            elif (attr_config.get('type') == 'list' and 
                  attr_config.get('item_type') == 'enum'):
                enum_class_name = attr_config.get('values')
                if enum_class_name:
                    try:
                        enum_class = getattr(enums_module, enum_class_name)
                        if issubclass(enum_class, Enum):
                            enum_mappings[field_name] = enum_class
                    except (AttributeError, TypeError):
                        logger.warning(
                            f"Could not find enum class '{enum_class_name}' "
                            f"for list field '{field_name}'"
                        )
        
        return enum_mappings

    @classmethod
    def register_product_enums(
        cls,
        product_type: str,
        enum_mappings: Dict[str, Type[Enum]]
    ) -> None:
        """Register enum mappings for a product type."""
        cls._enum_mappings[product_type] = {
            **cls._base_mappings,
            **enum_mappings
        }

    @classmethod
    def get_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
        """Get enum mappings for a product type."""
        if product_type not in cls._enum_mappings:
            return cls._base_mappings
        return cls._enum_mappings[product_type]

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of registered product types."""
        return list(cls._product_types)
        
    @classmethod
    def add_product_type(cls, product_type: str, enum_mappings: Dict[str, Type[Enum]]) -> None:
        """Add a new product type with its enum mappings."""
        cls._product_types.add(product_type)
        cls.register_product_enums(product_type, enum_mappings)


@lru_cache(maxsize=100)
def _read_and_encode_image(path: str) -> tuple:
    """Returns (base64_str, mime_type) for both APIs"""
    with open(path, "rb") as f:
        data = f.read()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return base64.b64encode(data).decode("utf-8"), mime_type


def create_image_message(image_data_url: str) -> Dict[str, Any]:
    """
    Create a message dictionary with image content for API requests.
    """
    return {
        "role": "user", 
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data_url.split(';')[0].split(':')[1],
                    "data": image_data_url.split(',')[1]
                }
            }
        ]
    }

def image_to_data_url(image_path: str) -> Optional[str]:
    """
    Convert an image file to a data URL format.
    Returns None if the file doesn't exist or conversion fails.
    """
    try:
        # Validate image path
        if not image_path:
            logger.error("Empty image path provided")
            return None
            
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
            
        if not path.is_file():
            logger.error(f"Path is not a file: {image_path}")
            return None
            
        # Check file size - warn if large
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 5:
            logger.warning(f"Image file is large ({file_size_mb:.2f}MB): {image_path}")
        
        # Determine MIME type based on file extension
        extension = path.suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }.get(extension)
        
        if not mime_type:
            # Use mimetypes library as fallback
            mime_type = mimetypes.guess_type(image_path)[0]
            
        if not mime_type:
            logger.warning(f"Could not determine MIME type for {image_path}, using image/jpeg")
            mime_type = 'image/jpeg'  # Default to JPEG
        
        # Read and encode file
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Format for Cohere API - full data URL
        data_url = f"data:{mime_type};base64,{img_data}"
        
        # Verify data URL format
        if not data_url.startswith("data:") or ";base64," not in data_url:
            logger.error(f"Generated invalid data URL format for {image_path}")
            return None
            
        logger.debug(f"Successfully converted image to data URL: {image_path}")
        return data_url
    except Exception as e:
        logger.exception(f"Error converting image to data URL: {str(e)}")
        return None