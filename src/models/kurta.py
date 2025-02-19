from typing import List, Optional, Any
from pydantic import Field, field_validator
import logging

from src.base import (
    ClothingItem, Closure, Fit, Neckline,
    Hemline, SleeveType, KurtaSet, Material
)
from src.utils.validation import find_best_enum_match

logger = logging.getLogger(__name__)


class Kurta(ClothingItem):
    # Design attributes
    sleeve_type: SleeveType
    neckline: Neckline
    closure: Closure
    fit: Fit
    hemline: Hemline
    kurta_set: KurtaSet
    
    # Physical attributes
    length: float = Field(..., gt=0)
    side_slits: bool = False
    
    # Optional bottom wear details
    bottom_material: Optional[Material] = None
    dupatta_material: Optional[Material] = None
    
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
                    f"Fuzzy matched sleeve_type '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return SleeveType(best_match.value)
        
        logger.warning(
            f"Could not match sleeve_type '{v}', defaulting to 'Others'"
        )
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
                return Neckline(best_match.value)
        
        logger.warning(
            f"Could not match neckline '{v}', defaulting to 'Others'"
        )
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
                return Closure(best_match.value)
        
        logger.warning(
            f"Could not match closure '{v}', defaulting to 'Others'"
        )
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
                return Fit(best_match.value)
        
        logger.warning(
            f"Could not match fit '{v}', defaulting to 'Others'"
        )
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
                return Hemline(best_match.value)
        
        logger.warning(
            f"Could not match hemline '{v}', defaulting to 'Others'"
        )
        return Hemline.OTHERS
    
    @field_validator('kurta_set', mode='before')
    def validate_kurta_set(cls, v: Any) -> KurtaSet:
        if isinstance(v, KurtaSet):
            return v
        
        if not v:
            return KurtaSet.OTHERS
            
        if isinstance(v, str):
            try:
                return KurtaSet(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, KurtaSet)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched kurta_set '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return KurtaSet(best_match.value)
        
        logger.warning(
            f"Could not match kurta_set '{v}', defaulting to 'Others'"
        )
        return KurtaSet.OTHERS
    
    
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
                "Fitted/form-fitting kurtas should not use pull-on closure"
            )
            
        # Validate hemline based on side slits
        if self.side_slits and self.hemline not in [
            Hemline.STRAIGHT, Hemline.CURVED
        ]:
            errors.append(
                "Side slits are only available for straight or curved hemlines"
            )
            
        # Validate bottom material for sets with bottom wear
        if self.kurta_set in [
            KurtaSet.KURTA_WITH_PALAZZO,
            KurtaSet.KURTA_WITH_PANTS,
            KurtaSet.COMPLETE_SET
        ] and not self.bottom_material:
            errors.append(
                "Bottom material must be specified for sets with bottom wear"
            )
            
        # Validate dupatta material for sets with dupatta
        if self.kurta_set in [
            KurtaSet.KURTA_WITH_DUPATTA,
            KurtaSet.COMPLETE_SET
        ] and not self.dupatta_material:
            errors.append(
                "Dupatta material must be specified for sets with dupatta"
            )
            
        return errors 