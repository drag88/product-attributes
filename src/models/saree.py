from typing import List, Optional, Set, Any, Union
from pydantic import field_validator, ValidationInfo, Field
import logging
from src.base import (
    ClothingItem, SareeType, BorderWidth, BorderDesign,
    PalluDesign, Material, EmbellishmentLevel
)
from src.utils.validation import validate_enum_field
from src.utils.config_loader import ConfigManager

logger = logging.getLogger(__name__)


class Saree(ClothingItem):
    # Load saree-specific configuration
    _config = ConfigManager.get_merged_config('saree')
    _attr_config = _config.get('attributes', {})
    
    # Design attributes
    saree_type: SareeType
    border_width: BorderWidth
    border_design: BorderDesign
    border_design_details: Optional[Set[str]] = Field(
        default=None,
        min_length=_attr_config.get(
            'border_design_details', {}
        ).get('min_length', 3),
        max_length=_attr_config.get(
            'border_design_details', {}
        ).get('max_length', 5)
    )
    pallu_design: PalluDesign
    
    # Physical attributes
    length: Optional[float] = Field(
        default=None,
        gt=_attr_config.get('length', {}).get('min', 0),
        lt=_attr_config.get('length', {}).get('max', 12)
    )
    width: Optional[float] = Field(
        default=None,
        gt=_attr_config.get('width', {}).get('min', 0.9),
        lt=_attr_config.get('width', {}).get('max', 1.4)
    )
    weight: Optional[int] = Field(
        default=None,
        gt=_attr_config.get('weight', {}).get('min', 300),
        lt=_attr_config.get('weight', {}).get('max', 1000)
    )
    
    # Additional features
    blouse_included: bool = Field(
        default=_attr_config.get('blouse_included', {}).get('default', False)
    )
    blouse_material: Optional[Material] = Field(
        default=None
    )
    pre_draped: bool = Field(
        default=_attr_config.get('pre_draped', {}).get('default', False)
    )
    reversible: bool = Field(
        default=_attr_config.get('reversible', {}).get('default', False)
    )

    @field_validator(
        'saree_type', 'border_width', 'border_design', 
        'pallu_design', 'blouse_material', 
        mode='before'
    )
    @classmethod
    def validate_enum_fields(
        cls, v: Any, info: ValidationInfo
    ) -> Union[SareeType, BorderWidth, BorderDesign, PalluDesign, Material]:
        field_name = info.field_name
        if not field_name:
            return v
            
        field_config = {
            'saree_type': (
                SareeType, 
                SareeType.OTHERS, 
                cls._attr_config.get('saree_type', {}).get('threshold', 0.7)
            ),
            'border_width': (
                BorderWidth, 
                BorderWidth.OTHERS, 
                cls._attr_config.get('border_width', {}).get('threshold', 0.7)
            ),
            'border_design': (
                BorderDesign, 
                BorderDesign.OTHERS, 
                cls._attr_config.get('border_design', {}).get('threshold', 0.7)
            ),
            'pallu_design': (
                PalluDesign, 
                PalluDesign.OTHERS, 
                cls._attr_config.get('pallu_design', {}).get('threshold', 0.7)
            ),
            'blouse_material': (
                Material, 
                Material.OTHERS, 
                cls._attr_config.get(
                    'blouse_material', {}
                ).get('threshold', 0.7),
                True
            )
        }
        
        if field_name not in field_config:
            return v
        
        config = field_config[field_name]
        enum_cls = config[0]
        default = config[1]
        threshold = config[2]
        allow_compound = len(config) > 3 and config[3]
        
        # Special case for blouse_material
        if field_name == 'blouse_material' and v is None:
            return None

        result = validate_enum_field(
            v,
            enum_cls,
            field_name,
            default=default,
            threshold=threshold,
            allow_compound=allow_compound
        )
        
        # Type cast to ensure correct return type
        if field_name == 'saree_type':
            return SareeType(str(result.value))
        elif field_name == 'border_width':
            return BorderWidth(str(result.value))
        elif field_name == 'border_design':
            return BorderDesign(result.value)
        elif field_name == 'pallu_design':
            return PalluDesign(result.value)
        elif field_name == 'blouse_material' and result is not None:
            return Material(result.value)
        
        return result

    @field_validator('saree_type', mode='before')
    @classmethod
    def validate_saree_type(cls, v: Any) -> SareeType:
        return validate_enum_field(
            v,
            SareeType,
            'saree_type',
            default=SareeType.OTHERS,
            threshold=cls._attr_config.get('saree_type', {}).get('threshold', 0.7)
        )

    def validate_attributes(self) -> List[str]:
        errors = []
        
        # Validate physical measurements
        min_length = self._attr_config.get('length', {}).get('min', 4.5)
        if self.length and self.length < min_length:
            errors.append("Saree length >= {min}m".format(min=min_length))
            
        min_width = self._attr_config.get('width', {}).get('min', 0.8)
        if self.width and self.width < min_width:
            errors.append(f"Saree width should be at least {min_width} meters")
            
        # Validate blouse material if included
        if self.blouse_included and not self.blouse_material:
            msg = "Blouse material must be specified when blouse is included"
            errors.append(msg)
            
        # Validate border design details if specified
        has_other_design = self.border_design != BorderDesign.OTHERS
        if has_other_design and self.border_design_details:
            msg = (
                "Border design details should only be provided "
                "for 'Others' design"
            )
            errors.append(msg)
            
        return errors 