from typing import Dict, Type, List, Optional, Any
from enum import Enum
from pathlib import Path
import yaml
import importlib
import logging
from src.base.clothing_item import ClothingItem
from src.base.utils import EnumRegistry
from src.utils.config_loader import ConfigManager
from src.models.saree import Saree

logger = logging.getLogger(__name__)

class ClothingFactory:
    """Factory for creating clothing items based on product type."""
    
    _registry: Dict[str, Type[ClothingItem]] = {}
    _enum_registry = EnumRegistry.get_instance()
    _config_manager = ConfigManager.get_instance()
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the factory by discovering and registering all product types."""
        try:
            # Clear existing registry to allow reinitialization
            cls._registry.clear()
            
            # Get available product types from config manager
            product_types = cls._config_manager.get_available_product_types()
            
            # Dynamically import and register product classes
            for product_type in product_types:
                cls._register_product_type(product_type)
                
            if not cls._registry:
                logger.warning("No product types were registered during initialization")
            else:
                logger.info(
                    f"Successfully registered {len(cls._registry)} product types: "
                    f"{list(cls._registry.keys())}"
                )
                
        except Exception as e:
            logger.error(f"Error during factory initialization: {str(e)}")
            raise
        
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
        logger.info(f"Enum mappings: {enum_mappings}")
        enum_type = enum_mappings.get(field_name)
        
        if enum_type:
            return enum_type
            
        # Fallback to config-based lookup if not found in registry
        if (attr_config.get('item_type') == 'enum' or 
            (attr_config.get('data_type') == 'list' and 
             attr_config.get('item_type') == 'enum')):
            
            # Import all enum classes
            enums_module = importlib.import_module('src.base.enums')
            enum_class_name = attr_config.get('allowed_values')
            
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
        cls._registry[product_type.lower()] = product_class
        logger.info(f"Registered product type: {product_type}")
    
    @classmethod
    def get_product_class(cls, product_type: str) -> Optional[Type[ClothingItem]]:
        """Get the class for a product type."""
        return cls._registry.get(product_type.lower())
    
    @classmethod
    def create_product(cls, product_type: str, **kwargs) -> Optional[ClothingItem]:
        """Create a product instance of the specified type."""
        if product_type in cls._registry:
            # Use specific product class if available
            product_class = cls._registry[product_type]
        else:
            # Fallback to base ClothingItem for undefined types
            logger.warning(f"Using base ClothingItem for undefined type: {product_type}")
            product_class = ClothingItem
            
        try:
            # Get product-specific config
            product_config = cls._config_manager.get_product_config(product_type)
            if not product_config:
                logger.error(f"No configuration found for {product_type}")
                return None
                
            # Create product instance
            return product_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Error creating {product_type}: {str(e)}")
            return None
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of registered product types."""
        return list(cls._registry.keys())
    
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