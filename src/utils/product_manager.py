from typing import Dict, Any, List, Optional, Type
import logging
import importlib
from pathlib import Path
import yaml
from src.factories.clothing_factory import ClothingFactory
from src.base.clothing_item import ClothingItem

logger = logging.getLogger(__name__)

class ProductManager:
    """Utility for managing product types."""
    
    @staticmethod
    def list_available_products() -> List[str]:
        """List all available product types."""
        factory = ClothingFactory.get_instance()
        return factory.get_available_types()
    
    @staticmethod
    def create_product_template(product_type: str) -> Dict[str, Any]:
        """Create a template configuration for a new product type."""
        # Start with a minimal template
        template = {
            "extends": "base_config.yaml",
            "attributes": {
                # Add product-specific attributes here
            }
        }
        
        return template
    
    @staticmethod
    def generate_product_class_template(product_type: str) -> str:
        """Generate template code for a new product class."""
        # Convert product_type to CamelCase for class name
        class_name = ''.join(word.capitalize() for word in product_type.split('_'))
        
        template = f"""from typing import List, Optional, Set, Any
from pydantic import Field, field_validator
import logging
from src.base import ClothingItem

logger = logging.getLogger(__name__)


class {class_name}(ClothingItem):
    \"\"\"
    {class_name} product class.
    Add product-specific attributes and validation logic here.
    \"\"\"
    
    # Add product-specific attributes here
    
    @field_validator('*')
    def validate_fields(cls, value, info):
        \"\"\"Validate all fields using common validation logic.\"\"\"
        return super().validate_fields(value, info)
    
    def validate_attributes(self) -> List[str]:
        \"\"\"Validate product-specific attributes.\"\"\"
        errors = []
        # Add product-specific validation logic here
        return errors
"""
        return template
    
    @staticmethod
    def add_new_product_type(
        product_type: str,
        config: Dict[str, Any],
        product_class_code: str
    ) -> bool:
        """Add a new product type with its configuration and class."""
        try:
            # 1. Create the product module file
            module_path = Path(__file__).parent.parent / 'models' / f"{product_type}.py"
            with open(module_path, 'w') as f:
                f.write(product_class_code)
                
            # 2. Import the new module and get the product class
            module_name = f"src.models.{product_type}"
            importlib.invalidate_caches()  # Clear import cache
            
            try:
                module = importlib.import_module(module_name)
                # Convert product_type to CamelCase for class name
                class_name = ''.join(word.capitalize() for word in product_type.split('_'))
                product_class = getattr(module, class_name)
                
                # 3. Register the new product type with the factory
                factory = ClothingFactory.get_instance()
                success = factory.add_product_type(
                    product_type, 
                    product_class,
                    config
                )
                
                if success:
                    logger.info(f"Successfully added new product type: {product_type}")
                    return True
                else:
                    logger.error(f"Failed to add product type: {product_type}")
                    return False
                    
            except (ImportError, AttributeError) as e:
                logger.error(f"Error importing new product module: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding new product type: {e}")
            return False
    
    @staticmethod
    def create_product(product_type: str, **kwargs) -> Optional[ClothingItem]:
        """Create a product instance of the specified type."""
        try:
            factory = ClothingFactory.get_instance()
            return factory.create(product_type, **kwargs)
        except Exception as e:
            logger.error(f"Error creating product: {e}")
            return None


# Example usage:
def add_dress_product_type():
    """Example of adding a new 'dress' product type."""
    # 1. Create a template configuration
    config = {
        "extends": "base_config.yaml",
        "attributes": {
            "dress_type": {
                "required": True,
                "in_search_context": True,
                "type": "enum",
                "values": "DressType"
            },
            "length": {
                "required": True,
                "in_search_context": True,
                "type": "enum",
                "values": "DressLength"
            },
            "sleeve_type": {
                "required": True,
                "in_search_context": True,
                "type": "enum",
                "values": "SleeveType"
            },
            "neckline": {
                "required": True,
                "in_search_context": True,
                "type": "enum",
                "values": "Neckline"
            },
            "waistline": {
                "required": True,
                "in_search_context": True,
                "type": "enum",
                "values": "Waistline"
            }
        }
    }
    
    # 2. Create a product class template
    product_class_code = """from typing import List, Optional, Set, Any
from pydantic import Field, field_validator
import logging
from src.base import (
    ClothingItem, DressType, DressLength, SleeveType, 
    Neckline, Waistline, Material
)

logger = logging.getLogger(__name__)


class Dress(ClothingItem):
    \"\"\"
    Dress product class.
    \"\"\"
    # Design attributes
    dress_type: DressType
    length: DressLength
    sleeve_type: SleeveType
    neckline: Neckline
    waistline: Waistline
    
    # Physical attributes
    material: Material
    
    def validate_attributes(self) -> List[str]:
        \"\"\"Validate product-specific attributes.\"\"\"
        errors = []
        
        # Add validation logic here
        # For example, certain dress types might require specific necklines
        
        return errors
"""
    
    # 3. Add the new product type
    success = ProductManager.add_new_product_type(
        'dress',
        config,
        product_class_code
    )
    
    return success 