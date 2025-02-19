from typing import Optional, Tuple, Type
from enum import Enum
import difflib


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