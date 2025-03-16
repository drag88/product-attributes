from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Union
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
        max_length=_attr_config.get('secondary_colors', {}).get(
            'max_length', 3
        )
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
        min_length=_attr_config.get('unique_design_element', {}).get(
            'min_length', 1
        ),
        max_length=_attr_config.get('unique_design_element', {}).get(
            'max_length', 3
        )
    )
    
    # Target audience
    gender: List[Gender] = Field(
        default_factory=lambda: [Gender.WOMEN]
    )
    age_group: List[AgeGroup] = Field(
        default_factory=lambda: [AgeGroup.ADULT]
    )
    
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
        description="Cohere embedding vector of search_context"
    )

    @classmethod
    def get_category_mapping(cls) -> tuple[list[tuple[re.Pattern, str]], str]:
        """Load category standardization rules from config"""
        config = ConfigManager.get_base_config()
        standardization_config = config.get('category_standardization', {})
        
        patterns = []
        for item in standardization_config.get('patterns', []):
            try:
                # Compile combined regex pattern
                combined_pattern = r'\b(?:{})\b'.format('|'.join(item['matches']))
                compiled_pattern = re.compile(combined_pattern, re.IGNORECASE)
                patterns.append((compiled_pattern, item['name']))
            except re.error as e:
                logger.error(f"Invalid regex pattern for {item['name']}: {e}")
                continue
                
        return (
            patterns,
            standardization_config.get('default_strategy', 'title_case')
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

    @field_validator(
        'primary_color', 'primary_color_detailed', 'material', 
        mode='before'
    )
    @classmethod
    def validate_single_enum_fields(cls, v: Any, info: ValidationInfo) -> Enum:
        # Default thresholds from config
        thresholds = {}
        
        # Get thresholds from config
        field_name = info.field_name
        if field_name:
            threshold = cls._attr_config.get(field_name, {}).get('threshold')
            if threshold is not None:
                thresholds[field_name] = threshold
        
        # Set default thresholds if not in config
        if field_name and field_name not in thresholds:
            default_thresholds = {
                'primary_color': 0.8,
                'primary_color_detailed': 0.8,
                'material': 0.8
            }
            thresholds[field_name] = default_thresholds.get(field_name, 0.8)
        
        field_config = {
            'primary_color': (
                Color, Color.OTHERS, thresholds.get('primary_color', 0.8)
            ),
            'primary_color_detailed': (
                ColorDetailed, 
                ColorDetailed.OTHERS, 
                thresholds.get('primary_color_detailed', 0.8)
            ),
            'material': (
                Material, 
                Material.OTHERS, 
                thresholds.get('material', 0.8), 
                True
            )
        }
        
        if field_name not in field_config:
            return v
            
        cls_name, default, threshold, *compound = field_config[field_name]
        return validate_enum_field(
            v, 
            cls_name, 
            field_name,
            default=default,
            threshold=threshold,
            allow_compound=bool(compound)
        )

    @field_validator(
        'secondary_colors', 'secondary_colors_detailed', 
        'occasions', 'style', 'gender', 'age_group', 
        mode='before'
    )
    @classmethod
    def validate_enum_lists(cls, v: Any, info: ValidationInfo) -> List[Enum]:
        # Get thresholds from config
        thresholds = {}
        
        # Get threshold from config for the current field
        field_name = info.field_name
        if field_name:
            threshold = cls._attr_config.get(field_name, {}).get('threshold')
            if threshold is not None:
                thresholds[field_name] = threshold
        
        # Set default thresholds if not in config
        if field_name and field_name not in thresholds:
            default_thresholds = {
                'secondary_colors': 0.8,
                'secondary_colors_detailed': 0.8,
                'occasions': 0.6,
                'style': 0.8,
                'gender': 0.9,
                'age_group': 0.9
            }
            thresholds[field_name] = default_thresholds.get(field_name, 0.8)
        
        field_config = {
            'secondary_colors': (
                Color, Color.OTHERS, thresholds.get('secondary_colors', 0.8)
            ),
            'secondary_colors_detailed': (
                ColorDetailed, ColorDetailed.OTHERS, 
                thresholds.get('secondary_colors_detailed', 0.8)
            ),
            'occasions': (
                Occasion, 
                Occasion.OTHERS, 
                thresholds.get('occasions', 0.6)
            ),
            'style': (Style, Style.OTHERS, thresholds.get('style', 0.7)),
            'gender': (Gender, Gender.WOMEN, thresholds.get('gender', 0.9)),
            'age_group': (
                AgeGroup, AgeGroup.ADULT, thresholds.get('age_group', 0.9)
            )
        }
        
        if field_name not in field_config:
            return v
            
        cls_name, default, threshold = field_config[field_name]
        return validate_enum_list(v, cls_name, field_name, default, threshold)

    @field_validator(
        'primary_color_hex', 'secondary_color_hexes', 
        'secondary_colors_detailed_hex', 'color_pairings', 
        mode='before'
    )
    @classmethod
    def validate_hex_codes(
        cls, v: Any, info: ValidationInfo
    ) -> Union[str, List[str]]:
        hex_regex = r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
        
        if info.field_name == 'primary_color_hex':
            if not re.match(hex_regex, v):
                raise ValueError(f"Invalid primary color hex code: {v}")
            return v
        
        if not isinstance(v, list):
            v = [v]
            
        for hex_code in v:
            if not re.match(hex_regex, hex_code):
                raise ValueError(f"Invalid hex code format: {hex_code}")
        
        return v

    @field_validator('embellishment_level', mode='before')
    @classmethod
    def validate_embellishment_level(cls, v: Any, info: ValidationInfo) -> Enum:
        return validate_enum_field(
            v, EmbellishmentLevel, 'embellishment_level',
            default=EmbellishmentLevel.NONE,
            threshold=cls._attr_config.get('embellishment_level', {}).get('threshold', 0.7)
        )

    @field_validator('embellishment', mode='before')
    @classmethod 
    def validate_embellishment_list(cls, v: Any, info: ValidationInfo) -> List[Enum]:
        return validate_enum_list(
            v, Embellishment, 'embellishment',
            default=Embellishment.NONE,
            threshold=cls._attr_config.get('embellishment', {}).get('threshold', 0.6)
        )

    @field_validator('pattern', mode='before')
    @classmethod
    def validate_patterns(cls, v: Any, info: ValidationInfo) -> List[Pattern]:
        return validate_enum_list(
            v,
            Pattern,
            'pattern',
            threshold=cls._attr_config.get('pattern', {}).get('threshold', 0.6)
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