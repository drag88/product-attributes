from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Union, Type
import re
import logging
from pydantic import (
    BaseModel, Field, field_validator, ValidationInfo
)
from .enums import (
    Color, ColorDetailed, Pattern, Material, 
    EmbellishmentLevel, Embellishment, Occasion, 
    Style, Gender, AgeGroup
)
from src.utils.validation import validate_enum_field, validate_enum_list
from src.utils.config_loader import ConfigManager
import pandas as pd

logger = logging.getLogger(__name__)


class ClothingItem(BaseModel, ABC):
    # Load base configuration
    _config = ConfigManager.get_base_config()
    _attr_config = _config.get('attributes', {})
    
    # Basic attributes with defaults from config
    brand: str = Field(
        default=_attr_config.get('brand', {}).get('default', '')
    )
    brand_title: str
    title: str
    description: str
    price: float = Field(
        ..., 
        gt=_attr_config.get('price', {}).get('min', 0)
    )
    
    # Color attributes
    primary_color: Color
    primary_color_detailed: ColorDetailed
    primary_color_hex: str = Field(
        ..., 
        pattern=r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
    )
    secondary_colors: List[Color] = Field(
        default_factory=list,
        max_length=_attr_config.get('secondary_colors', {}).get('max_length', 3)
    )
    secondary_color_hexes: List[str] = Field(default_factory=list)
    secondary_colors_detailed: List[ColorDetailed] = Field(
        default_factory=list,
        max_length=_attr_config.get('secondary_colors_detailed', {}).get(
            'max_length', 3
        )
    )
    secondary_colors_detailed_hex: List[str] = Field(default_factory=list)
    color_pairings: List[str] = Field(default_factory=list)
    
    # Design attributes
    pattern: List[Pattern] = Field(default_factory=list)
    material: Material
    embellishment_level: EmbellishmentLevel
    embellishment: List[Embellishment] = Field(
        default_factory=lambda: ["None"],
        max_length=_attr_config.get('embellishment', {}).get('max_length', 5)
    )
    embellishment_detailed: List[str] = Field(
        default_factory=lambda: ["None"]
    )
    
    # Usage attributes
    occasions: List[Occasion] = Field(
        default_factory=list,
        max_length=_attr_config.get('occasions', {}).get('max_length', 3)
    )
    occasions_detailed: List[str] = Field(default_factory=list)
    style: List[Style] = Field(default_factory=list)
    
    # Product details
    care_instructions: str
    size: List[str]
    unique_design_element: List[str] = Field(
        default_factory=list,
        min_length=_attr_config.get('unique_design_element', {}).get('min_length', 1),
        max_length=_attr_config.get('unique_design_element', {}).get('max_length', 3)
    )
    
    # Target audience
    gender: List[Gender] = Field(default_factory=lambda: [Gender.WOMEN])
    age_group: List[AgeGroup] = Field(default_factory=lambda: [AgeGroup.ADULT])
    
    # Additional info
    coordinating_items: Dict = Field(
        default_factory=lambda: {
            "clothing": [],
            "accessories": [],
            "footwear": [],
            "additional_apparel": [],
            "styling_suggestions": []
        }
    )
    
    text_embedding: List[float] = Field(
        default_factory=list,
        description="Cohere text embedding vector of search_context"
    )

    image_embedding: List[float] = Field(
        default_factory=list,
        description="Cohere image embedding vector of search_context"
    )

    @classmethod
    def _get_field_config(cls) -> Dict[str, tuple[Type[Enum], Enum, float]]:
        """Get field configurations for validation."""
        return {
            'primary_color': (Color, Color.OTHERS, 0.8),
            'primary_color_detailed': (ColorDetailed, ColorDetailed.OTHERS, 0.8),
            'material': (Material, Material.OTHERS, 0.8),
            'embellishment_level': (EmbellishmentLevel, EmbellishmentLevel.NONE, 0.7),
            'pattern': (Pattern, None, 0.6),
            'occasions': (Occasion, Occasion.OTHERS, 0.6),
            'style': (Style, Style.OTHERS, 0.7),
            'gender': (Gender, Gender.WOMEN, 0.9),
            'age_group': (AgeGroup, AgeGroup.ADULT, 0.9),
            'secondary_colors': (Color, Color.OTHERS, 0.8),
            'secondary_colors_detailed': (ColorDetailed, ColorDetailed.OTHERS, 0.8),
            'embellishment': (Embellishment, Embellishment.NONE, 0.6)
        }

    @field_validator(
        'primary_color', 'primary_color_detailed', 'material',
        'embellishment_level', 'pattern', 'occasions', 'style',
        'gender', 'age_group', 'secondary_colors', 
        'secondary_colors_detailed', 'embellishment',
        mode='before'
    )
    @classmethod
    def validate_fields(cls, v: Any, info: ValidationInfo) -> Any:
        field_name = info.field_name
        if not field_name:
            return v
            
        field_config = cls._get_field_config()
        if field_name not in field_config:
            return v
            
        enum_cls, default, threshold = field_config[field_name]
        
        # Handle list fields
        if field_name in {
            'pattern', 'occasions', 'style', 'gender', 'age_group',
            'secondary_colors', 'secondary_colors_detailed', 'embellishment'
        }:
            return validate_enum_list(
                v, enum_cls, field_name, default=default, threshold=threshold
            )
            
        # Handle single enum fields
        return validate_enum_field(
            v, enum_cls, field_name, default=default, threshold=threshold
        )

    @classmethod
    def standardize_category(cls, category: str) -> str:
        """Standardize product category using configurable rules"""
        if pd.isna(category) or not isinstance(category, str):
            return "Uncategorized"
            
        category = category.strip()
        patterns, default_strategy = cls.get_category_mapping()
        
        # Check against all patterns
        for pattern, standardized_name in patterns:
            if pattern.search(category):
                return standardized_name
                
        # Apply default strategy
        if default_strategy == "title_case":
            return category.title()
        elif default_strategy == "upper_case":
            return category.upper()
        elif default_strategy == "lower_case":
            return category.lower()
        else:
            return category

    @classmethod
    def get_category_mapping(cls) -> tuple[list[tuple[re.Pattern, str]], str]:
        """Load category standardization rules from config"""
        config = ConfigManager.get_base_config()
        standardization_config = config.get('category_standardization', {})
        
        patterns = []
        for item in standardization_config.get('patterns', []):
            try:
                combined_pattern = r'\b(?:{})\b'.format('|'.join(item['matches']))
                compiled_pattern = re.compile(combined_pattern, re.IGNORECASE)
                patterns.append((compiled_pattern, item['name'].title()))
            except re.error as e:
                logger.error(f"Invalid regex pattern for {item['name']}: {e}")
                continue
                
        return (
            patterns,
            standardization_config.get('default_strategy', 'title_case')
        )

    def validate_attributes(self) -> List[str]:
        """Validate base attributes"""
        errors = []
        
        # Validate required fields
        required_attrs = self._attr_config.keys()
        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                errors.append(f"Missing required attribute: {attr}")
                
        return errors 