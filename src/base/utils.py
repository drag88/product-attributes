from enum import Enum
from typing import Type, Optional, Tuple, Dict, List, Set, Any
import re
import difflib
import logging
from functools import lru_cache
from src.base.enums import (
    Color, ColorDetailed, Pattern, Material, 
    EmbellishmentLevel, Embellishment, Occasion, 
    Style, Gender, AgeGroup
)
import base64
from pathlib import Path
# from src.base.clothing_item import ClothingItem

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())


def get_word_variations(word: str) -> Set[str]:
    """Generate common variations of a word."""
    word = word.lower()
    variations = {word}
    
    suffix_map = {
        'ed': '', 'ing': '', 'tion': 't', 
        'sion': 's', 'al': '', 'ic': '',
        'y': '', 'ies': 'y', 'ied': 'y'
    }
    
    for suffix, replacement in suffix_map.items():
        if word.endswith(suffix):
            variations.add(word[:-len(suffix)] + replacement)
    
    return variations


@lru_cache(maxsize=100)
def build_enum_variations(enum_class: Type[Enum]) -> Dict[str, Enum]:
    """Build cached mapping of normalized variations to enum values."""
    variations = {}
    
    for enum_value in enum_class:
        base_value = enum_value.value
        words = normalize_text(base_value).split()
        
        variations[normalize_text(base_value)] = enum_value
        
        for word in words:
            word_vars = get_word_variations(word)
            for var in word_vars:
                variations[var] = enum_value
                
        if len(words) > 1:
            for word in words:
                variations[normalize_text(word)] = enum_value
    
    return variations


def get_similarity_score(text1: str, text2: str) -> float:
    """Get similarity score between two texts."""
    return difflib.SequenceMatcher(
        None, 
        normalize_text(text1), 
        normalize_text(text2)
    ).ratio()


def find_best_enum_match(
    value: str,
    enum_class: Type[Enum],
    threshold: float = 0.8
) -> Tuple[Optional[Enum], float]:
    """Find best matching enum value using multiple strategies."""
    if not value:
        return None, 0.0
        
    normalized_value = normalize_text(value)
    variations = build_enum_variations(enum_class)
    
    # Try exact matches first
    if normalized_value in variations:
        return variations[normalized_value], 1.0
    
    # Try individual words for compound values
    words = normalized_value.split()
    if len(words) > 1:
        for word in words:
            if word in variations:
                return variations[word], 0.9
    
    # Try fuzzy matching as last resort
    best_score = 0.0
    best_match = None
    
    for enum_value in enum_class:
        score = get_similarity_score(value, enum_value.value)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = enum_value
            
    return best_match, best_score


def get_default_enum_value(enum_class: Type[Enum]) -> Enum:
    for default_name in ['OTHERS', 'OTHER', 'NONE']:
        try:
            return enum_class[default_name]
        except KeyError:
            continue
    return next(iter(enum_class))


def validate_enum_value(
    value: Optional[str], 
    enum_class: Type[Enum],
    field_name: str
) -> Enum:
    if not value:
        return get_default_enum_value(enum_class)
        
    if isinstance(value, str):
        try:
            return enum_class(value)
        except ValueError:
            pass
            
        best_match, score = find_best_enum_match(value, enum_class)
        if best_match and score >= 0.8:
            msg = (
                f"Fuzzy matched {field_name} '{value}' to "
                f"'{best_match.value}' with score {score}"
            )
            logger.info(msg)
            return best_match
    
    msg = f"Could not match {field_name} '{value}', using default"
    logger.warning(msg)
    return get_default_enum_value(enum_class)


def validate_enum_list(
    values: Optional[List[str]], 
    enum_class: Type[Enum],
    field_name: str
) -> List[Enum]:
    if not values:
        return []
        
    result = []
    for value in values:
        result.append(validate_enum_value(value, enum_class, field_name))
            
    return result or [get_default_enum_value(enum_class)]


class EnumRegistry:
    _instance = None
    _enum_mappings: Dict[str, Dict[str, Type[Enum]]] = {}
    _base_mappings: Dict[str, Type[Enum]] = {
        'primary_color': Color,
        'primary_color_detailed': ColorDetailed,
        'secondary_colors': Color,
        'secondary_colors_detailed': ColorDetailed,
        'pattern': Pattern,
        'material': Material,
        'embellishment_level': EmbellishmentLevel,
        'embellishment': Embellishment,
        'occasions': Occasion,
        'style': Style,
        'gender': Gender,
        'age_group': AgeGroup
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_product_enums(
        cls,
        product_type: str,
        enum_mappings: Dict[str, Type[Enum]]
    ) -> None:
        """Register enum mappings for a product type."""
        cls._enum_mappings[product_type] = {
            **cls._base_mappings,
            **enum_mappings
        }

    @classmethod
    def get_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
        """Get enum mappings for a product type."""
        if product_type not in cls._enum_mappings:
            return cls._base_mappings
        return cls._enum_mappings[product_type]

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of registered product types."""
        return list(cls._enum_mappings.keys())


# class ClothingFactory:
#     _registry: Dict[str, Type[ClothingItem]] = {}
#     _enum_registry = EnumRegistry()

#     @classmethod
#     def register(
#         cls, 
#         product_type: str, 
#         product_class: Type[ClothingItem],
#         enum_mappings: Optional[Dict[str, Type[Enum]]] = None
#     ) -> None:
#         """Register a product type with its class and enum mappings."""
#         cls._registry[product_type] = product_class
#         if enum_mappings:
#             cls._enum_registry.register_product_enums(
#                 product_type, 
#                 enum_mappings
#             )

#     @classmethod
#     def get_available_types(cls) -> List[str]:
#         """Get list of registered product types."""
#         return list(cls._registry.keys())

#     @classmethod
#     def get_enum_mappings(cls, product_type: str) -> Dict[str, Type[Enum]]:
#         """Get enum mappings for a product type."""
#         return cls._enum_registry.get_enum_mappings(product_type)

#     @classmethod
#     def create(cls, product_type: str, **kwargs) -> ClothingItem:
#         """Create a product instance with validated attributes."""
#         if product_type not in cls._registry:
#             raise ValueError(f"Unknown product type: {product_type}")
            
#         product_class = cls._registry[product_type]
#         return product_class(**kwargs) 

def create_image_message(image_path: str) -> Dict[str, Any]:
    """Create message payload for image processing."""
    data_url = image_to_data_url(image_path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": f"image/{Path(image_path).suffix[1:]}",
            "data": data_url.split(",")[1]
        }
    }

def image_to_data_url(image_path: str) -> str:
    """Convert image to data URL format."""
    with open(image_path, "rb") as f:
        data = f.read()
        file_type = Path(image_path).suffix[1:]
        base64_str = base64.b64encode(data).decode("utf-8")
        return f"data:image/{file_type};base64,{base64_str}" 