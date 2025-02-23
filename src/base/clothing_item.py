from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Union
import re
import logging
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
from .enums import (
    Color, ColorDetailed, Pattern, Material, 
    EmbellishmentLevel, Embellishment, Occasion, 
    Style, Gender, AgeGroup
)
from src.utils.validation import find_best_enum_match, validate_enum_field, validate_enum_list

logger = logging.getLogger(__name__)

class ClothingItem(BaseModel, ABC):
    brand: str = "suta"
    brand_title: str
    title: str
    description: str
    price: float = Field(..., gt=0)
    
    # Color attributes
    primary_color: Color
    primary_color_detailed: ColorDetailed
    primary_color_hex: str = Field(
        ..., 
        pattern=r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
    )
    secondary_colors: List[Color] = Field(default_factory=list)
    secondary_color_hexes: List[str] = Field(default_factory=list)
    secondary_colors_detailed: List[ColorDetailed] = Field(
        default_factory=list
    )
    secondary_colors_detailed_hex: List[str] = Field(default_factory=list)
    color_pairings: List[str] = Field(default_factory=list)
    
    # Design attributes
    pattern: List[Pattern] = Field(default_factory=list)
    material: Material
    embellishment_level: EmbellishmentLevel
    embellishment: List[Embellishment] = Field(default_factory=lambda: ["None"])
    embellishment_detailed: List[str] = Field(default_factory=lambda: ["None"])
    
    # Usage attributes
    occasions: List[Occasion] = Field(default_factory=list)
    occasion_detailed: List[str] = Field(default_factory=list)
    style: List[Style] = Field(default_factory=list)
    
    # Product details
    care_instructions: str
    size: List[str]
    unique_design_element: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=3
    )
    
    # Target audience
    gender: List[Gender] = Field(default_factory=lambda: [Gender.WOMEN])
    age_group: List[AgeGroup] = Field(default_factory=lambda: [AgeGroup.ADULT])
    
    # Additional info
    coordinating_items: Dict = Field(default_factory=lambda: {
        "clothing": [],
        "accessories": [],
        "footwear": [],
        "additional_apparel": [],
        "styling_suggestions": []
    })
    
    # Search context
    search_context: str = Field(
        default="",
        description="Concatenated attributes for semantic search"
    )
    
    text_embedding: List[float] = Field(
        default_factory=list,
        description="Cohere embedding vector of search_context"
    )

    @field_validator('primary_color', 'primary_color_detailed', 'material', 
                    'embellishment_level', mode='before')
    def validate_single_enum_fields(cls, v: Any, info: ValidationInfo) -> Enum:
        field_config = {
            'primary_color': (Color, Color.OTHERS, 0.8),
            'primary_color_detailed': (ColorDetailed, ColorDetailed.OTHERS, 0.8),
            'material': (Material, Material.OTHERS, 0.7, True),
            'embellishment_level': (EmbellishmentLevel, EmbellishmentLevel.NONE, 0.7)
        }
        
        cls_name, default, threshold, *compound = field_config[info.field_name]
        return validate_enum_field(
            v, 
            cls_name, 
            info.field_name,
            default=default,
            threshold=threshold,
            allow_compound=bool(compound)
        )

    @field_validator('secondary_colors', 'secondary_colors_detailed', 'occasions',
                    'style', 'gender', 'age_group', 'embellishment', mode='before')
    def validate_enum_lists(cls, v: Any, info: ValidationInfo) -> List[Enum]:
        field_config = {
            'secondary_colors': (Color, Color.OTHERS, 0.8),
            'secondary_colors_detailed': (ColorDetailed, ColorDetailed.OTHERS, 0.8),
            'occasions': (Occasion, Occasion.OTHERS, 0.7),
            'style': (Style, Style.OTHERS, 0.7),
            'gender': (Gender, Gender.WOMEN, 0.9),
            'age_group': (AgeGroup, AgeGroup.ADULT, 0.9),
            'embellishment': (Embellishment, Embellishment.NONE, 0.7)
        }
        
        cls_name, default, threshold = field_config[info.field_name]
        return validate_enum_list(v, cls_name, info.field_name, default, threshold)

    @field_validator('primary_color_hex', 'secondary_color_hexes', 
                    'secondary_colors_detailed_hex', 'color_pairings', mode='before')
    def validate_hex_codes(cls, v: Any, info: ValidationInfo) -> Union[str, List[str]]:
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

    @model_validator(mode='after')
    def build_search_context(self) -> Any:
        secondary_colors_str = ', '.join(
            c.value for c in self.secondary_colors
        ) if self.secondary_colors else ''
        patterns_str = ', '.join(
            p.value for p in self.pattern
        ) if self.pattern else ''
        occasions_str = ', '.join(
            o.value for o in self.occasions
        ) if self.occasions else ''
        styles_str = ', '.join(
            s.value for s in self.style
        ) if self.style else ''
        embellishments_str = ', '.join(
            e.value for e in self.embellishment
        ) if self.embellishment else ''
        
        context_parts = [
            f"Colors: {self.primary_color.value}" + (
                f", {secondary_colors_str}" if secondary_colors_str else ""
            ),
            f"Materials: {self.material.value}",
            f"Patterns: {patterns_str}" if patterns_str else None,
            f"Occasions: {occasions_str}" if occasions_str else None,
            f"Styles: {styles_str}" if styles_str else None,
            f"Embellishments: {self.embellishment_level.value}" + (
                f" {embellishments_str}" if embellishments_str else ""
            ),
            f"Unique features: {', '.join(self.unique_design_element)}" if self.unique_design_element else None
        ]
        
        self.search_context = ' | '.join(
            part for part in context_parts if part is not None
        )
        return self

    @abstractmethod
    def validate_attributes(self) -> List[str]:
        """Validate product-specific attributes"""
        pass 