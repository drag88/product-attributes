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
    @classmethod
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
    @classmethod
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
    @classmethod
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
    @classmethod
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
    @classmethod
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
    @classmethod
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
        
        # Validate physical measurements
        if self.length < 24:
            errors.append("Kurta length should be at least 24 inches")
            
        # Validate bottom material if kurta set includes bottom
        bottom_sets = [
            KurtaSet.KURTA_PANT, KurtaSet.KURTA_PAJAMA,
            KurtaSet.KURTA_DHOTI, KurtaSet.KURTA_SALWAR
        ]
        if self.kurta_set in bottom_sets and not self.bottom_material:
            msg = "Bottom material must be specified for kurta sets with bottoms"
            errors.append(msg)
            
        # Validate dupatta material if kurta set includes dupatta
        dupatta_sets = [
            KurtaSet.KURTA_DUPATTA, KurtaSet.KURTA_PANT_DUPATTA,
            KurtaSet.KURTA_PAJAMA_DUPATTA, KurtaSet.KURTA_SALWAR_DUPATTA
        ]
        if self.kurta_set in dupatta_sets and not self.dupatta_material:
            msg = "Dupatta material must be specified for kurta sets with dupatta"
            errors.append(msg)
            
        return errors 