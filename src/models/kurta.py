from typing import List, Optional, Any, Dict, Type
from pydantic import Field, field_validator, ValidationInfo
import logging
from importlib import import_module
from src.base import (
    ClothingItem, Closure, Fit, Neckline,
    Hemline, SleeveType, KurtaSet, Material
)
from src.utils.validation import find_best_enum_match, validate_enum_field
from src.utils.config_loader import ConfigManager

logger = logging.getLogger(__name__)


class Kurta(ClothingItem):
    # Load kurta-specific configuration
    _config = ConfigManager.get_merged_config('kurta')
    _attr_config = _config.get('attributes', {})
    
    @classmethod
    def _get_enum_class(cls, enum_name: str) -> Type:
        """Dynamically import and return enum class from src.base."""
        try:
            return getattr(import_module('src.base'), enum_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import enum {enum_name}: {e}")
            raise ValueError(f"Invalid enum class: {enum_name}")

    @classmethod
    def _get_field_config(cls) -> Dict[str, tuple[Type, Any, float]]:
        """Dynamically generate field configurations from product config."""
        field_config = {}
        
        for field_name, config in cls._attr_config.items():
            if config.get('item_type') == 'enum':
                enum_class_name = config.get('allowed_values')
                if enum_class_name:
                    try:
                        enum_cls = cls._get_enum_class(enum_class_name)
                        field_config[field_name] = (
                            enum_cls,
                            getattr(enum_cls, 'OTHERS', None),
                            config.get('threshold', 0.8)
                        )
                    except ValueError:
                        continue
        
        return field_config

    def __init_subclass__(cls):
        """Dynamically add fields based on configuration."""
        super().__init_subclass__()
        
        # Add fields based on configuration
        for field_name, config in cls._attr_config.items():
            field_type: Any = str  # Default type
            field_default = None
            field_kwargs = {}
            
            if config.get('item_type') == 'enum':
                try:
                    field_type = cls._get_enum_class(config['allowed_values'])
                    if config.get('default'):
                        field_default = getattr(field_type, config['default'])
                except ValueError:
                    continue
            
            elif config.get('item_type') == 'boolean':
                field_type = bool
                field_default = config.get('default', False)
            
            elif config.get('item_type') == 'string':
                field_type = str
                field_default = config.get('default', '')
                if 'min_length' in config:
                    field_kwargs['min_length'] = config['min_length']
                if 'max_length' in config:
                    field_kwargs['max_length'] = config['max_length']
            
            elif config.get('item_type') == 'float':
                field_type = float
                field_default = config.get('default')
                if 'min' in config:
                    field_kwargs['gt'] = config['min']
                if 'max' in config:
                    field_kwargs['lt'] = config['max']
            
            # Add the field to the class
            if field_default is not None:
                setattr(cls, field_name, Field(
                    default=field_default, **field_kwargs
                ))
            else:
                setattr(cls, field_name, field_type)

    @field_validator('*', mode='before')
    @classmethod
    def validate_fields(
        cls, v: Any, info: ValidationInfo
    ) -> Any:
        field_name = info.field_name
        if not field_name:
            return v
            
        field_config = cls._get_field_config()
        if field_name in field_config:
            config = field_config[field_name]
            enum_cls = config[0]
            default = config[1]
            threshold = config[2]
            
            result = validate_enum_field(
                v,
                enum_cls,
                field_name,
                default=default,
                threshold=threshold
            )
            
            return enum_cls(str(result.value))
            
        return v

    def validate_attributes(self) -> List[str]:
        """Validate attribute combinations based on business rules."""
        errors = []
        field_config = self._get_field_config()
        
        # Get all enum fields and their values
        enum_fields = {
            field: getattr(self, field)
            for field in field_config.keys()
            if hasattr(self, field)
        }
        
        # Validate physical measurements
        length = getattr(self, 'length', None)
        if length is not None and length < 24:
            errors.append("Kurta length should be at least 24 inches")
            
        # Get KurtaSet enum class
        kurta_set_cls = self._get_enum_class('KurtaSet')
        kurta_set = enum_fields.get('kurta_set')
        
        # Validate bottom material if kurta set includes bottom
        bottom_sets = [
            kurta_set_cls.KURTA_WITH_PANTS,
            kurta_set_cls.KURTA_WITH_PALAZZO,
            kurta_set_cls.COMPLETE_SET
        ]
        if kurta_set in bottom_sets and not getattr(self, 'bottom_material', None):
            msg = "Bottom material must be specified for kurta sets with bottoms"
            errors.append(msg)
            
        # Validate dupatta material if kurta set includes dupatta
        dupatta_sets = [
            kurta_set_cls.KURTA_WITH_DUPATTA,
            kurta_set_cls.COMPLETE_SET
        ]
        if kurta_set in dupatta_sets and not getattr(self, 'dupatta_material', None):
            msg = "Dupatta material must be specified for kurta sets with dupatta"
            errors.append(msg)
            
        # Validate required string fields
        for field_name, config in self._attr_config.items():
            if (config.get('required', False) and 
                    config.get('item_type') == 'string' and
                    not getattr(self, field_name, None)):
                errors.append(f"{field_name} is required")
            
        return errors 