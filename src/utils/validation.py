from typing import Optional, Tuple, Type, Any, List
from enum import Enum
import difflib
import logging

logger = logging.getLogger(__name__)


def find_best_enum_match(
    value: str,
    enum_class: Type[Enum],
    threshold: float = 0.8
) -> Tuple[Optional[Enum], float]:
    """Find the best matching enum value using fuzzy string matching.
    
    Args:
        value: The string value to match
        enum_class: The enum class to match against
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Tuple of (best matching enum member, similarity score) or (None, 0.0)
    """
    if not value:
        return None, 0.0
        
    # Normalize input
    value = value.lower().strip()
    
    # Try exact match first
    for member in enum_class:
        if member.value.lower() == value:
            return member, 1.0
    
    # Try fuzzy matching
    matches = []
    for member in enum_class:
        score = difflib.SequenceMatcher(
            None, 
            value,
            member.value.lower()
        ).ratio()
        if score >= threshold:
            matches.append((member, score))
    
    if matches:
        return max(matches, key=lambda x: x[1])
        
    return None, 0.0 

def validate_enum_field(
    value: Any,
    enum_cls: Type[Enum],
    field_name: str,
    default: Enum,
    threshold: float = 0.7,
    allow_compound: bool = False
) -> Enum:
    if isinstance(value, enum_cls):
        return value
    
    if not value:
        return default
    
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
        
        best_match, score = find_best_enum_match(value, enum_cls)
        if best_match and score >= threshold:
            logger.info(f"Matched {field_name} '{value}' to '{best_match.value}' (score: {score})")
            return best_match
        
        if allow_compound:
            for word in value.split():
                word_match, word_score = find_best_enum_match(word, enum_cls)
                if word_match and word_score >= 0.9:
                    logger.info(f"Matched compound {field_name} word '{word}' from '{value}'")
                    return word_match
    
    logger.warning(f"No match for {field_name} '{value}', using default")
    return default

def validate_enum_list(
    values: Any,
    enum_cls: Type[Enum],
    field_name: str,
    default: Enum,
    threshold: float = 0.7
) -> List[Enum]:
    if not values:
        return [default]
    
    validated = []
    for value in (values if isinstance(values, list) else [values]):
        validated.append(
            validate_enum_field(value, enum_cls, field_name, default, threshold)
        )
    return validated 