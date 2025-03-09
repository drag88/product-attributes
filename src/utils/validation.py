from typing import Optional, Tuple, Type, Any, List, Dict, Set
from enum import Enum
import difflib
import logging
import re
from functools import lru_cache

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
    """Find best matching enum value using multiple strategies.
    
    Args:
        value: The string value to match
        enum_class: The enum class to match against
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Tuple of (best matching enum member, similarity score) or (None, 0.0)
    """
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
    """Get default enum value, trying common default names first."""
    for default_name in ['OTHERS', 'OTHER', 'NONE']:
        try:
            return enum_class[default_name]
        except KeyError:
            continue
    return next(iter(enum_class))


def validate_enum_field(
    value: Any,
    enum_cls: Type[Enum],
    field_name: str,
    default: Optional[Enum] = None,
    threshold: float = 0.7,
    allow_compound: bool = True
) -> Enum:
    """Validate and convert a value to an enum member.
    
    Args:
        value: The value to validate
        enum_cls: The enum class to validate against
        field_name: Name of the field (for logging)
        default: Default enum value if no match found
        threshold: Minimum similarity score for fuzzy matching
        allow_compound: Whether to try matching individual words
        
    Returns:
        Matched enum member or default
    """
    if isinstance(value, enum_cls):
        return value
    
    if not value:
        return default if default is not None else get_default_enum_value(enum_cls)
    
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
        
        best_match, score = find_best_enum_match(value, enum_cls)
        if best_match and score >= threshold:
            logger.info(
                f"Matched {field_name} '{value}' to '{best_match.value}' "
                f"(score: {score})"
            )
            return best_match
    
    logger.warning(f"No match for {field_name} '{value}', using default")
    return default if default is not None else get_default_enum_value(enum_cls)


def validate_enum_list(
    values: Any,
    enum_cls: Type[Enum],
    field_name: str,
    default: Optional[Enum] = None,
    threshold: float = 0.7
) -> List[Enum]:
    """Validate and convert a list of values to enum members.
    
    Args:
        values: List of values to validate
        enum_cls: The enum class to validate against
        field_name: Name of the field (for logging)
        default: Default enum value if no match found
        threshold: Minimum similarity score for fuzzy matching
        
    Returns:
        List of matched enum members
    """
    if not values:
        default_value = default if default is not None else get_default_enum_value(enum_cls)
        return [default_value]
    
    validated = []
    for value in (values if isinstance(values, list) else [values]):
        validated.append(
            validate_enum_field(
                value, enum_cls, field_name, default, threshold
            )
        )
    return validated 