from typing import Dict, Type, List, Optional
from enum import Enum
from pathlib import Path
import yaml
from src.base.clothing_item import ClothingItem
from src.models.saree import Saree
from src.models.blouse import Blouse
from src.models.kurta import Kurta
from src.base.enums import *  # We'll get all enums dynamically


class ClothingFactory:
    _registry: Dict[str, Type[ClothingItem]] = {}
    _enum_mappings: Dict[str, Dict[str, Type[Enum]]] = {}
    _config_cache: Dict[str, dict] = {}

    @classmethod
    def _load_config(cls, product_type: str) -> dict:
        """Load config for a product type."""
        if product_type not in cls._config_cache:
            config_path = Path(__file__).parent.parent.parent / 'config' / f'{product_type}_config.yaml'
            with open(config_path) as f:
                cls._config_cache[product_type] = yaml.safe_load(f)
        return cls._config_cache[product_type]

    @classmethod
    def _get_enum_type(cls, field_name: str, validation_rule: dict) -> Optional[Type[Enum]]:
        """Get enum type from validation rule."""
        if validation_rule.get('type') in ['enum', 'list']:
            enum_name = validation_rule.get('values')
            return globals().get(enum_name)
        return None

    @classmethod
    def _build_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
        """Build enum mappings from config validation rules."""
        config = cls._load_config(product_type)
        mappings = {}
        
        validation_rules = config.get('validation_rules', {})
        for field_name, rule in validation_rules.items():
            enum_type = cls._get_enum_type(field_name, rule)
            if enum_type:
                mappings[field_name] = enum_type
                
                # Handle list types that might have enum items
                if rule.get('type') == 'list' and rule.get('item_type') == 'enum':
                    mappings[f'{field_name}_item'] = enum_type
        
        return mappings

    @classmethod
    def register(
        cls, 
        product_type: str, 
        product_class: Type[ClothingItem],
        enum_mappings: Optional[Dict[str, Type[Enum]]] = None
    ):
        """Register a product type with its class and enum mappings."""
        cls._registry[product_type] = product_class
        
        # Use provided mappings or build from config
        if enum_mappings is None:
            enum_mappings = cls._build_enum_mappings(product_type)
        cls._enum_mappings[product_type] = enum_mappings

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of registered product types."""
        return list(cls._registry.keys())

    @classmethod
    def get_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
        """Get enum mappings for a product type."""
        if product_type not in cls._enum_mappings:
            cls._enum_mappings[product_type] = cls._build_enum_mappings(product_type)
        return cls._enum_mappings[product_type]

    @classmethod
    def create(cls, product_type: str, **kwargs) -> ClothingItem:
        """Create a product instance with validated attributes."""
        if product_type not in cls._registry:
            raise ValueError(f"Unknown product type: {product_type}")
            
        product_class = cls._registry[product_type]
        return product_class(**kwargs)


# Register product types
for product_type, product_class in [
    ('saree', Saree),
    ('blouse', Blouse),
    ('kurta', Kurta)
]:
    ClothingFactory.register(product_type, product_class) 