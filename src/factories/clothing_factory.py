from typing import Dict, Type, List, Optional, Any
from enum import Enum
from pathlib import Path
import yaml
import importlib
import logging
from src.base import ClothingItem
from src.base.utils import EnumRegistry
from src.utils.config_loader import ProductConfigManager
from src.models.saree import Saree

logger = logging.getLogger(__name__)

class ClothingFactory:
    """Factory for creating clothing items based on product type."""
    
    _registry: Dict[str, Type[ClothingItem]] = {}
    _enum_registry = EnumRegistry.get_instance()
    _config_manager = ProductConfigManager.get_instance()
    
    @classmethod
    def initialize(cls):
        """Initialize the factory by discovering and registering all product types."""
        # Clear existing registry to allow reinitialization
        cls._registry.clear()
        
        # Get available product types from config manager
        product_types = cls._config_manager.get_available_product_types()
        
        # Dynamically import and register product classes
        for product_type in product_types:
            cls._register_product_type(product_type)
            
        logger.info(f"Registered product types: {list(cls._registry.keys())}")
        
    @classmethod
    def _register_product_type(cls, product_type: str) -> None:
        """Dynamically import and register a product class."""
        try:
            # Convert product_type to CamelCase for class name
            class_name = ''.join(word.capitalize() for word in product_type.split('_'))
            
            # Try to import the product class from models module
            module_name = f"src.models.{product_type}"
            try:
                module = importlib.import_module(module_name)
                product_class = getattr(module, class_name)
                
                # Register the product class
                cls.register(product_type, product_class)
                logger.info(f"Successfully registered product type: {product_type}")
                
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Could not import product class for {product_type}: {e}"
                )
                
        except Exception as e:
            logger.error(f"Error registering product type {product_type}: {e}")

    @classmethod
    def _get_enum_type(cls, field_name: str, attr_config: dict, 
                      product_type: str) -> Optional[Type[Enum]]:
        """Get enum type from attribute config."""
        # First try to get from EnumRegistry
        enum_mappings = cls._enum_registry.get_enum_mappings(product_type)
        enum_type = enum_mappings.get(field_name)
        
        if enum_type:
            return enum_type
            
        # Fallback to config-based lookup if not found in registry
        if (attr_config.get('type') == 'enum' or 
            (attr_config.get('type') == 'list' and 
             attr_config.get('item_type') == 'enum')):
            
            # Import all enum classes
            enums_module = importlib.import_module('src.base.enums')
            enum_class_name = attr_config.get('values')
            
            if enum_class_name:
                try:
                    return getattr(enums_module, enum_class_name)
                except AttributeError:
                    logger.warning(
                        f"Enum class {enum_class_name} not found for {field_name}"
                    )
                    
        return None

    @classmethod
    def register(cls, product_type: str, product_class: Type[ClothingItem]) -> None:
        """Register a product type with its corresponding class."""
        cls._registry[product_type] = product_class
        logger.info(f"Registered product type: {product_type}")
    
    @classmethod
    def create(cls, product_type: str, data: Dict[str, Any]) -> Optional[ClothingItem]:
        """Create a clothing item of the specified type."""
        if product_type not in cls._registry:
            logger.error(f"Unknown product type: {product_type}")
            return None
        
        try:
            product_class = cls._registry[product_type]
            return product_class(**data)
        except Exception as e:
            logger.error(f"Error creating {product_type}: {str(e)}")
            return None
    
    @classmethod
    def get_available_types(cls) -> Dict[str, Type[ClothingItem]]:
        """Get all registered product types."""
        return cls._registry.copy()
    
    @classmethod
    def get_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
        """Get enum mappings for a product type."""
        return cls._enum_registry.get_enum_mappings(product_type)
        
    @classmethod
    def get_product_config(cls, product_type: str) -> Dict[str, Any]:
        """Get configuration for a product type."""
        return cls._config_manager.get_merged_config(product_type)

    @classmethod
    def add_product_type(cls, 
                        product_type: str, 
                        product_class: Type[ClothingItem],
                        config: Dict[str, Any]) -> bool:
        """Add a new product type with its class and configuration."""
        try:
            # Add to config manager
            success = cls._config_manager.add_product_type(product_type, config)
            if not success:
                return False
                
            # Register the product class
            cls.register(product_type, product_class)
            
            # Extract and register enum mappings
            attributes = config.get('attributes', {})
            enum_mappings = cls._enum_registry._extract_enum_mappings_from_attributes(
                attributes
            )
            cls._enum_registry.add_product_type(product_type, enum_mappings)
            
            logger.info(f"Successfully added new product type: {product_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product type {product_type}: {e}")
            return False

# Remove the automatic initialization
# Initialize the factory
# ClothingFactory.initialize() 