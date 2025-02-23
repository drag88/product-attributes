from typing import List, Optional, Any
from pydantic import Field, field_validator
from pydantic._internal import _validators
import logging
from src.base import (
    ClothingItem, SareeType, BorderWidth,
    BorderDesign, PalluDesign, Material
)
from src.utils.validation import find_best_enum_match, validate_enum_field

logger = logging.getLogger(__name__)


class Saree(ClothingItem):
    # Saree-specific attributes
    saree_type: SareeType
    border_width: BorderWidth
    border_design: BorderDesign
    border_design_details: List[str] = Field(default_factory=list)
    pallu_design: PalluDesign
    pre_draped: bool = False
    reversible: bool = False
    
    # Physical attributes
    length: Optional[float] = Field(..., gt=0)
    width: Optional[float] = Field(..., gt=0)
    weight: Optional[float] = Field(..., gt=0)
    
    # Blouse details
    blouse_included: bool = False
    blouse_material: Optional[Material] = None

    @field_validator('saree_type', 'border_width', 'border_design', 
                    'pallu_design', 'blouse_material', mode='before')
    def validate_saree_fields(cls, v: Any, info) -> Any:
        field_config = {
            'saree_type': (SareeType, SareeType.OTHERS, 0.7),
            'border_width': (BorderWidth, BorderWidth.OTHERS, 0.7),
            'border_design': (BorderDesign, BorderDesign.OTHERS, 0.7),
            'pallu_design': (PalluDesign, PalluDesign.OTHERS, 0.7),
            'blouse_material': (Material, Material.OTHERS, 0.7, True)
        }
        
        if info.field_name not in field_config:
            return v
        
        config = field_config[info.field_name]
        enum_cls = config[0]
        default = config[1]
        threshold = config[2]
        allow_compound = len(config) > 3 and config[3]
        
        if info.field_name == 'blouse_material' and v is None:
            return None

        return validate_enum_field(
            v,
            enum_cls,
            info.field_name,
            default=default,
            threshold=threshold,
            allow_compound=allow_compound
        )

    def validate_attributes(self) -> List[str]:
        errors = []
        
        # Validate physical measurements
        if self.length and self.length < 4.5:
            errors.append("Saree length should be at least 4.5 meters")
        if self.width and self.width < 0.8:
            errors.append("Saree width should be at least 0.8 meters")
            
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