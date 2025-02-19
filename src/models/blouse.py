from typing import List, Any
from pydantic import field_validator
import logging
from src.base import (
    ClothingItem, Closure, Fit, Neckline,
    Hemline, SleeveType
)
from src.utils.validation import find_best_enum_match

logger = logging.getLogger(__name__)


class Blouse(ClothingItem):
    # Design attributes
    sleeve_type: SleeveType
    neckline: Neckline
    closure: Closure
    fit: Fit
    hemline: Hemline
    
    # Physical attributes
    padded: bool = False

    @field_validator('sleeve_type', mode='before')
    def validate_sleeve_type(cls, v: Any) -> SleeveType:
        if isinstance(v, SleeveType):
            return v
        
        if not v:
            return SleeveType.OTHERS
            
        if isinstance(v, str):
            try:
                return SleeveType(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, SleeveType)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched sleeve type '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match sleeve type '{v}', defaulting to 'Others'")
        return SleeveType.OTHERS

    @field_validator('neckline', mode='before')
    def validate_neckline(cls, v: Any) -> Neckline:
        if isinstance(v, Neckline):
            return v
        
        if not v:
            return Neckline.OTHERS
            
        if isinstance(v, str):
            try:
                return Neckline(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Neckline)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched neckline '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match neckline '{v}', defaulting to 'Others'")
        return Neckline.OTHERS

    @field_validator('closure', mode='before')
    def validate_closure(cls, v: Any) -> Closure:
        if isinstance(v, Closure):
            return v
        
        if not v:
            return Closure.OTHERS
            
        if isinstance(v, str):
            try:
                return Closure(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Closure)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched closure '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match closure '{v}', defaulting to 'Others'")
        return Closure.OTHERS

    @field_validator('fit', mode='before')
    def validate_fit(cls, v: Any) -> Fit:
        if isinstance(v, Fit):
            return v
        
        if not v:
            return Fit.OTHERS
            
        if isinstance(v, str):
            try:
                return Fit(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Fit)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched fit '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match fit '{v}', defaulting to 'Others'")
        return Fit.OTHERS

    @field_validator('hemline', mode='before')
    def validate_hemline(cls, v: Any) -> Hemline:
        if isinstance(v, Hemline):
            return v
        
        if not v:
            return Hemline.OTHERS
            
        if isinstance(v, str):
            try:
                return Hemline(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Hemline)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched hemline '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return best_match
        
        logger.warning(f"Could not match hemline '{v}', defaulting to 'Others'")
        return Hemline.OTHERS
    
    def validate_attributes(self) -> List[str]:
        errors = []
        
        # Validate sleeve type and neckline combination
        if (self.sleeve_type == SleeveType.OFF_SHOULDER and 
                self.neckline != Neckline.OFF_SHOULDER):
            errors.append(
                "Off-shoulder sleeve type requires off-shoulder neckline"
            )
            
        # Validate closure based on fit
        if (self.fit in [Fit.FITTED, Fit.FORM_FITTING] and 
                self.closure == Closure.PULL_ON):
            errors.append(
                "Fitted/form-fitting blouses should not use pull-on closure"
            )
            
        # Validate hemline based on fit
        if self.fit == Fit.CROP and self.hemline != Hemline.CROPPED:
            errors.append("Crop fit requires cropped hemline")
            
        return errors 