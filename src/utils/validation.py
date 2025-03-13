from typing import Optional, Tuple, Type, Any, List, Dict, Set
from enum import Enum
import difflib
import logging
import re
from functools import lru_cache
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Enhanced normalization preserving compound terms"""
    # Keep hyphens and apostrophes but replace other special chars
    text = re.sub(r"[^a-zA-Z0-9'’-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("-", " ")  # Treat hyphens as spaces for matching


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
    """Weighted similarity score favoring partial matches"""
    seq = SequenceMatcher(None, text1, text2)
    ratio = seq.ratio()
    
    # Partial ratio calculation from V1
    len1, len2 = len(text1), len(text2)
    partial_ratio = 0.0
    if len1 <= len2:
        partial_ratio = SequenceMatcher(None, text1, text2[:len1]).ratio()
    else:
        partial_ratio = SequenceMatcher(None, text1[:len2], text2).ratio()
    
    # Weighted average favoring partial matches
    return (ratio * 0.3) + (partial_ratio * 0.7)


def find_best_enum_match(
    value: str,
    enum_class: Type[Enum],
    threshold: float = 0.6
) -> Tuple[Optional[Enum], float]:
    normalized_value = normalize_text(value)
    enum_variations = build_enum_variations(enum_class)
    
    # Check for direct match first
    if normalized_value in enum_variations:
        return enum_variations[normalized_value], 1.0
    
    best_match = None
    best_score = 0.0
    
    # Enhanced substring matching with boosted scores
    for variation, enum_member in enum_variations.items():
        if normalized_value in variation:
            # Boost score for contained matches
            score = min(0.8 + (len(normalized_value)/len(variation)) * 0.2, 1.0)
            if score > best_score:
                best_score = score
                best_match = enum_member
        elif variation in normalized_value:
            score = 0.7  # Base score for reverse containment
            if score > best_score:
                best_score = score
                best_match = enum_member
    
    if best_score >= threshold:
        return best_match, best_score
    
    # Proceed with fuzzy matching if no substring match
    for variation, enum_member in enum_variations.items():
        score = get_similarity_score(normalized_value, variation)
        if score > best_score:
            best_score = score
            best_match = enum_member
    
    return (best_match, best_score) if best_score >= threshold else (None, 0.0)


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
    threshold: float = 0.8,
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
    
    if not value or (isinstance(value, str) and not value.strip()):
        default_value = default if default is not None else get_default_enum_value(enum_cls)
        logger.debug(f"Empty value for {field_name}, using default: {default_value.value}")
        return default_value
    
    if isinstance(value, str):
        # Try exact match first
        try:
            return enum_cls(value)
        except ValueError:
            pass
            
        # Try fuzzy matching
        best_match, score = find_best_enum_match(value, enum_cls)
        if best_match and score >= threshold:  # Strictly respect threshold
            logger.info(
                f"Matched {field_name} '{value}' to '{best_match.value}' "
                f"(score: {score})"
            )
            return best_match
            
        # If threshold not met, try compound word matching if allowed
        if allow_compound and ' ' in value:
            words = value.split()
            for word in words:
                best_match, score = find_best_enum_match(word, enum_cls)
                if best_match and score >= threshold:  # Strictly respect threshold
                    logger.info(
                        f"Matched {field_name} word '{word}' to "
                        f"'{best_match.value}' (score: {score})"
                    )
                    return best_match
    
    # If we get here, try a lower threshold as a fallback
    if isinstance(value, str):
        fallback_threshold = max(threshold - 0.2, 0.5)  # Lower threshold but not below 0.5
        best_match, score = find_best_enum_match(value, enum_cls, threshold=fallback_threshold)
        if best_match and score >= fallback_threshold:
            logger.info(
                f"Fallback match for {field_name} '{value}' to '{best_match.value}' "
                f"(score: {score})"
            )
            return best_match
    
    logger.warning(
        f"No match for {field_name} '{value}' with threshold {threshold}, "
        "using default"
    )
    return default if default is not None else get_default_enum_value(enum_cls)


def validate_enum_list(
    values: Any,
    enum_cls: Type[Enum],
    field_name: str,
    default: Optional[Enum] = None,
    threshold: float = 0.8
) -> List[Enum]:
    """Validate a list of values against an enum class.
    
    Args:
        values: List of values to validate
        enum_cls: The enum class to validate against
        field_name: Name of the field (for logging)
        default: Default enum value if no match found
        threshold: Minimum similarity score for fuzzy matching
        
    Returns:
        List of validated enum members
    """
    if not values:
        # Return empty list instead of list with default value
        return []
        
    if not isinstance(values, list):
        values = [values]
        
    result = []
    for value in values:
        validated = validate_enum_field(
            value,
            enum_cls,
            field_name,
            default=default,
            threshold=threshold  # Pass through threshold
        )
        if validated is not None:
            result.append(validated)
            
    # Only return default if result is empty and default is provided
    if not result and default:
        return [default]
    return result 