from typing import List, Optional, Any
from pydantic import Field, field_validator
import logging
from src.base import (
    ClothingItem, SareeType, BorderWidth,
    BorderDesign, PalluDesign, Material
)
from src.utils.validation import find_best_enum_match

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

    @field_validator('saree_type', mode='before')
    def validate_saree_type(cls, v: Any) -> SareeType:
        if isinstance(v, SareeType):
            return v
        
        if not v:
            return SareeType.OTHERS
            
        if isinstance(v, str):
            try:
                return SareeType(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, SareeType)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched saree type '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match saree type '{v}', defaulting to 'Others'")
        return SareeType.OTHERS

    @field_validator('border_width', mode='before')
    def validate_border_width(cls, v: Any) -> BorderWidth:
        if isinstance(v, BorderWidth):
            return v
        
        if not v:
            return BorderWidth.OTHERS
            
        if isinstance(v, str):
            try:
                return BorderWidth(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, BorderWidth)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched border width '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match border width '{v}', defaulting to 'Others'")
        return BorderWidth.OTHERS

    @field_validator('border_design', mode='before')
    def validate_border_design(cls, v: Any) -> BorderDesign:
        if isinstance(v, BorderDesign):
            return v
        
        if not v:
            return BorderDesign.OTHERS
            
        if isinstance(v, str):
            try:
                return BorderDesign(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, BorderDesign)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched border design '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match border design '{v}', defaulting to 'Others'")
        return BorderDesign.OTHERS

    @field_validator('pallu_design', mode='before')
    def validate_pallu_design(cls, v: Any) -> PalluDesign:
        if isinstance(v, PalluDesign):
            return v
        
        if not v:
            return PalluDesign.OTHERS
            
        if isinstance(v, str):
            try:
                return PalluDesign(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, PalluDesign)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched pallu design '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match pallu design '{v}', defaulting to 'Others'")
        return PalluDesign.OTHERS

    @field_validator('blouse_material', mode='before')
    def validate_blouse_material(cls, v: Any) -> Optional[Material]:
        if v is None:
            return None
            
        if isinstance(v, Material):
            return v
            
        if isinstance(v, str):
            try:
                return Material(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Material)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched blouse material '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match blouse material '{v}', defaulting to 'Others'")
        return Material.OTHERS
    
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