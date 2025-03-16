import logging
from enum import Enum
from typing import Dict, Type, Set
from unittest.mock import patch, MagicMock

from src.base.utils import EnumRegistry
from src.utils.config_loader import ConfigManager
from src.base.enums import (
    Color, Material, Pattern, Style,
    Occasion, Gender, AgeGroup
)

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_registry(registry: EnumRegistry) -> None:
    """Helper function to inspect registry state"""
    logger.info("=== Registry State ===")
    logger.info(f"Product Types: {sorted(list(registry._product_types))}")
    
    logger.info("\nBase Mappings:")
    for field, enum_class in registry._base_mappings.items():
        logger.info(f"  {field}: {enum_class.__name__}")
        values = [e.value for e in enum_class]
        logger.info(f"    First 5 values: {values[:5]}")
    
    logger.info("\nEnum Mappings by Product:")
    for product, mappings in registry._enum_mappings.items():
        logger.info(f"\n  {product}:")
        for field, enum_class in mappings.items():
            logger.info(f"    {field}: {enum_class.__name__}")
            values = [e.value for e in enum_class]
            logger.info(f"      First 5 values: {values[:5]}")
    
    logger.info("=" * 50)


def reset_registry():
    """Reset the registry's state for testing"""
    EnumRegistry._instance = None
    EnumRegistry._enum_mappings = {}
    EnumRegistry._base_mappings = {}
    EnumRegistry._product_types = set()
    EnumRegistry._initialized = False


class MockConfigManager:
    def __init__(self, base_config: Dict, product_configs: Dict):
        self._base_config = base_config
        self._product_configs = product_configs
        self._instance = None

    def get_base_config(self) -> Dict:
        logger.debug(f"Getting base config: {self._base_config}")
        return self._base_config

    def get_product_config(self, product_type: str) -> Dict:
        config = self._product_configs.get(product_type, {})
        logger.debug(f"Getting config for {product_type}: {config}")
        return config

    def get_available_product_types(self) -> Set[str]:
        return set(self._product_configs.keys())


def setup_test_config():
    """Set up test configuration for the registry"""
    logger.info("Setting up test configuration")
    
    # Reset registry
    reset_registry()
    
    # Add some base attributes that should use enums
    base_config = {
        'attributes': {
            'color': {'type': 'enum', 'values': 'Color'},
            'material': {'type': 'enum', 'values': 'Material'},
            'style': {'type': 'enum', 'values': 'Style'},
            'occasion': {'type': 'enum', 'values': 'Occasion'},
            'gender': {'type': 'enum', 'values': 'Gender'}
        }
    }
    
    # Add product-specific attributes
    product_configs = {
        'saree': {
            'attributes': {
                'border_pattern': {'type': 'enum', 'values': 'Pattern'},
                'age_group': {'type': 'enum', 'values': 'AgeGroup'}
            }
        },
        'kurta': {
            'attributes': {
                'pattern': {'type': 'enum', 'values': 'Pattern'},
                'age_group': {'type': 'enum', 'values': 'AgeGroup'}
            }
        }
    }
    
    return base_config, product_configs


@patch('src.base.utils.ConfigManager')
def test_enum_registry(mock_config_class):
    """Test the EnumRegistry functionality"""
    logger.info("Starting EnumRegistry test")
    
    # 0. Setup test configuration
    logger.info("\nStep 0: Setting up test configuration")
    base_config, product_configs = setup_test_config()
    
    logger.debug("Base config being used:")
    logger.debug(base_config)
    logger.debug("Product configs being used:")
    logger.debug(product_configs)
    
    # Create mock config manager
    mock_cm = MockConfigManager(base_config, product_configs)
    
    # Set up the mock class (this is what patch provides)
    mock_config_class.get_instance.return_value = mock_cm
    mock_config_class.get_base_config.return_value = base_config
    mock_config_class.get_product_config.side_effect = mock_cm.get_product_config
    mock_config_class.get_available_product_types.return_value = set(product_configs.keys())
    
    # 1. Get registry instance
    logger.info("\nStep 1: Getting EnumRegistry instance")
    registry = EnumRegistry.get_instance()
    inspect_registry(registry)
    
    # 2. Get available product types
    logger.info("\nStep 2: Getting available product types")
    product_types = registry.get_available_types()
    logger.info(f"Available product types: {sorted(product_types)}")
    
    # 3. Test enum mappings for each product type
    logger.info("\nStep 3: Testing enum mappings for each product type")
    for product_type in sorted(product_types):
        logger.info(f"\nTesting mappings for: {product_type}")
        mappings = registry.get_enum_mappings(product_type)
        
        logger.info("Field to Enum mappings:")
        for field_name, enum_class in mappings.items():
            try:
                enum_values = [e.value for e in enum_class]
                logger.info(f"  {field_name}: {enum_class.__name__}")
                logger.info(f"    Values: {enum_values[:5]}...")
            except Exception as e:
                logger.error(f"Error processing enum {field_name}: {e}")
    
    # 4. Test adding a new product type
    logger.info("\nStep 4: Testing adding new product type")
    try:
        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"
        
        test_mappings: Dict[str, Type[Enum]] = {
            "test_field": TestEnum
        }
        
        registry.add_product_type("test_product", test_mappings)
        logger.info("Added test product type")
        inspect_registry(registry)
        
    except Exception as e:
        logger.error(f"Error adding test product type: {e}")


if __name__ == "__main__":
    try:
        test_enum_registry()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True) 