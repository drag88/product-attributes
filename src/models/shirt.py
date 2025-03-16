from typing import List, Any, Dict, Type
from pydantic import field_validator, ValidationInfo, Field
import logging
from importlib import import_module
from src.base import ClothingItem
from src.utils.validation import validate_enum_field
from src.utils.config_loader import ConfigManager

logger = logging.getLogger(__name__)


class Shirt(ClothingItem):
    _config = ConfigManager.get_merged_config('shirt')
    _attr_config = _config.get('attributes', {})
    
    @classmethod
    def _get_enum_class(cls, enum_name: str) -> Type:
        try:
            return getattr(import_module('src.base'), enum_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import enum {enum_name}: {e}")
            raise ValueError(f"Invalid enum class: {enum_name}")

    @classmethod
    def _get_field_config(cls) -> Dict[str, tuple[Type, Any, float]]:
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
        super().__init_subclass__()
        
        for field_name, config in cls._attr_config.items():
            field_type: Any = str
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
        errors = []
        enum_fields = {
            field: getattr(self, field)
            for field in self._get_field_config().keys()
            if hasattr(self, field)
        }
        
        # Validate shirt type and collar combination
        if (enum_fields.get('shirt_type') == 
                self._get_enum_class('ShirtType').FORMAL and 
                enum_fields.get('collar') == 
                self._get_enum_class('ShirtCollar').BAND):
            errors.append("Formal shirts should not have band collars")
            
        # Validate fit and weave pattern combination
        if (enum_fields.get('fit') == 
                self._get_enum_class('ShirtFit').SLIM and 
                enum_fields.get('weave_pattern') == 
                self._get_enum_class('ShirtWeave').LOOSE):
            errors.append(
                "Slim fit shirts should not have loose weave patterns"
            )
        
        # Validate required string fields
        for field_name, config in self._attr_config.items():
            if (config.get('required', False) and 
                    config.get('item_type') == 'string' and
                    not getattr(self, field_name, None)):
                errors.append(f"{field_name} is required")
            
        return errors 