from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re
import logging
from pydantic import BaseModel, Field, field_validator, model_validator
from .enums import (
    Color, ColorDetailed, Pattern, Material, 
    EmbellishmentLevel, Embellishment, Occasion, 
    Style, Gender, AgeGroup
)
from src.utils.validation import find_best_enum_match

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
    text_embedding: List[float] = Field(default_factory=list)

    @field_validator('primary_color', mode='before')
    def validate_primary_color(cls, v: Any) -> Color:
        if isinstance(v, Color):
            return v
        
        if not v:
            return Color.OTHERS
            
        if isinstance(v, str):
            try:
                return Color(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Color)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched primary color '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return Color(best_match.value)
        
        logger.warning(
            f"Could not match primary color '{v}', defaulting to 'Others'"
        )
        return Color.OTHERS

    @field_validator('primary_color_detailed', mode='before')
    def validate_primary_color_detailed(cls, v: Any) -> ColorDetailed:
        if isinstance(v, ColorDetailed):
            return v
        
        if not v:
            return ColorDetailed.OTHERS
            
        if isinstance(v, str):
            try:
                return ColorDetailed(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, ColorDetailed)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched detailed color '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return ColorDetailed(best_match.value)
        
        msg = (
            f"Could not match detailed color '{v}', "
            "defaulting to 'Others'"
        )
        logger.warning(msg)
        return ColorDetailed.OTHERS

    @field_validator('secondary_colors', mode='before')
    def validate_secondary_colors(cls, v: Any) -> List[Color]:
        if not v:
            return []
            
        if not isinstance(v, list):
            v = [v]
            
        validated_colors = []
        for color in v:
            if isinstance(color, Color):
                validated_colors.append(color)
            elif isinstance(color, str):
                best_match, score = find_best_enum_match(color, Color)
                if best_match and score >= 0.8:
                    msg = (
                        f"Fuzzy matched secondary color '{color}' to "
                        f"'{best_match.value}' with score {score}"
                    )
                    logger.info(msg)
                    validated_colors.append(best_match)
                else:
                    logger.warning(
                        f"Could not match secondary color '{color}', "
                        "defaulting to 'Others'"
                    )
                    validated_colors.append(Color.OTHERS)
            else:
                logger.warning("Invalid secondary color type, defaulting to 'Others'")
                validated_colors.append(Color.OTHERS)
                
        return validated_colors

    @field_validator('secondary_colors_detailed', mode='before')
    def validate_secondary_colors_detailed(cls, v: Any) -> List[ColorDetailed]:
        if not v:
            return []
            
        if not isinstance(v, list):
            v = [v]
            
        validated_colors = []
        for color in v:
            if isinstance(color, ColorDetailed):
                validated_colors.append(color)
            elif isinstance(color, str):
                best_match, score = find_best_enum_match(color, ColorDetailed)
                if best_match and score >= 0.8:
                    msg = (
                        f"Fuzzy matched detailed color '{color}' to "
                        f"'{best_match.value}' with score {score}"
                    )
                    logger.info(msg)
                    validated_colors.append(best_match)
                else:
                    msg = (
                        f"Could not match detailed color '{color}', "
                        "defaulting to 'Others'"
                    )
                    logger.warning(msg)
                    validated_colors.append(ColorDetailed.OTHERS)
            else:
                msg = "Invalid detailed color type, defaulting to 'Others'"
                logger.warning(msg)
                validated_colors.append(ColorDetailed.OTHERS)
                
        return validated_colors

    @field_validator('pattern', mode='before')
    def validate_pattern(cls, v: Any) -> List[Pattern]:
        if not v:
            return [Pattern.OTHERS]
            
        if not isinstance(v, list):
            v = [v]
            
        validated_patterns = []
        for pattern in v:
            if isinstance(pattern, Pattern):
                validated_patterns.append(pattern)
            elif isinstance(pattern, str):
                best_match, score = find_best_enum_match(pattern, Pattern)
                if best_match and score >= 0.8:
                    validated_patterns.append(best_match)
                else:
                    msg = (
                        f"Invalid pattern '{pattern}' → "
                        "defaulting to 'Others'"
                    )
                    logger.warning(msg)
                    validated_patterns.append(Pattern.OTHERS)
            else:
                msg = "Invalid pattern type → defaulting to 'Others'"
                logger.warning(msg)
                validated_patterns.append(Pattern.OTHERS)
        
        return validated_patterns or [Pattern.OTHERS]

    @field_validator('material', mode='before')
    def validate_material(cls, v: Any) -> Material:
        if isinstance(v, Material):
            return v
        
        if not v:
            return Material.OTHERS
            
        if isinstance(v, str):
            try:
                return Material(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, Material)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched material '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return Material(best_match.value)
        
        msg = f"Could not match material '{v}', defaulting to 'Others'"
        logger.warning(msg)
        return Material.OTHERS

    @field_validator('embellishment_level', mode='before')
    def validate_embellishment_level(cls, v: Any) -> EmbellishmentLevel:
        if isinstance(v, EmbellishmentLevel):
            return v
        
        if not v:
            return EmbellishmentLevel.NONE
            
        if isinstance(v, str):
            try:
                return EmbellishmentLevel(v)
            except ValueError:
                pass
                
            best_match, score = find_best_enum_match(v, EmbellishmentLevel)
            if best_match and score >= 0.8:
                msg = (
                    f"Fuzzy matched embellishment level '{v}' to "
                    f"'{best_match.value}' with score {score}"
                )
                logger.info(msg)
                return EmbellishmentLevel(best_match.value)
        
        msg = (
            f"Could not match embellishment level '{v}', "
            "defaulting to 'None'"
        )
        logger.warning(msg)
        return EmbellishmentLevel.NONE

    @field_validator('embellishment', mode='before')
    def validate_embellishment(cls, v: Any) -> List[Embellishment]:
        if not v:
            return [Embellishment.NONE]
            
        if not isinstance(v, list):
            v = [v]
            
        validated_embellishments = []
        for embellishment in v:
            if isinstance(embellishment, Embellishment):
                validated_embellishments.append(embellishment)
            elif isinstance(embellishment, str):
                best_match, score = find_best_enum_match(embellishment, Embellishment)
                if best_match and score >= 0.8:
                    logger.info(f"Fuzzy matched embellishment '{embellishment}' to '{best_match.value}' with score {score}")
                    validated_embellishments.append(Embellishment(best_match.value))
                else:
                    logger.warning(f"Could not match embellishment '{embellishment}', defaulting to 'None'")
                    validated_embellishments.append(Embellishment.NONE)
            else:
                logger.warning(f"Invalid embellishment type, defaulting to 'None'")
                validated_embellishments.append(Embellishment.NONE)
                
        return validated_embellishments or [Embellishment.NONE]

    @field_validator('occasions', mode='before')
    def validate_occasions(cls, v: Any) -> List[Occasion]:
        if not v:
            return []
            
        if not isinstance(v, list):
            v = [v]
            
        validated_occasions = []
        for occasion in v:
            if isinstance(occasion, Occasion):
                validated_occasions.append(occasion)
            elif isinstance(occasion, str):
                best_match, score = find_best_enum_match(occasion, Occasion)
                if best_match and score >= 0.8:
                    logger.info(f"Fuzzy matched occasion '{occasion}' to '{best_match.value}' with score {score}")
                    validated_occasions.append(best_match)
                else:
                    logger.warning(f"Could not match occasion '{occasion}', defaulting to 'Others'")
                    validated_occasions.append(Occasion.OTHERS)
            else:
                logger.warning(f"Invalid occasion type, defaulting to 'Others'")
                validated_occasions.append(Occasion.OTHERS)
                
        return validated_occasions

    @field_validator('style', mode='before')
    def validate_style(cls, v: Any) -> List[Style]:
        if not v:
            return []
            
        if not isinstance(v, list):
            v = [v]
            
        validated_styles = []
        for style in v:
            if isinstance(style, Style):
                validated_styles.append(style)
            elif isinstance(style, str):
                best_match, score = find_best_enum_match(style, Style)
                if best_match and score >= 0.8:
                    logger.info(f"Fuzzy matched style '{style}' to '{best_match.value}' with score {score}")
                    validated_styles.append(best_match)
                else:
                    logger.warning(f"Could not match style '{style}', defaulting to 'Others'")
                    validated_styles.append(Style.OTHERS)
            else:
                logger.warning(f"Invalid style type, defaulting to 'Others'")
                validated_styles.append(Style.OTHERS)
                
        return validated_styles

    @field_validator('gender', mode='before')
    def validate_gender(cls, v: Any) -> List[Gender]:
        if not v:
            return [Gender.WOMEN]
            
        if not isinstance(v, list):
            v = [v]
            
        validated_genders = []
        for gender in v:
            if isinstance(gender, Gender):
                validated_genders.append(gender)
            elif isinstance(gender, str):
                best_match, score = find_best_enum_match(gender, Gender)
                if best_match and score >= 0.8:
                    logger.info(f"Fuzzy matched gender '{gender}' to '{best_match.value}' with score {score}")
                    validated_genders.append(best_match)
                else:
                    logger.warning(f"Could not match gender '{gender}', defaulting to 'Women'")
                    validated_genders.append(Gender.WOMEN)
            else:
                logger.warning(f"Invalid gender type, defaulting to 'Women'")
                validated_genders.append(Gender.WOMEN)
                
        return validated_genders or [Gender.WOMEN]

    @field_validator('age_group', mode='before')
    def validate_age_group(cls, v: Any) -> List[AgeGroup]:
        if not v:
            return [AgeGroup.ADULT]
            
        if not isinstance(v, list):
            v = [v]
            
        validated_age_groups = []
        for age_group in v:
            if isinstance(age_group, AgeGroup):
                validated_age_groups.append(age_group)
            elif isinstance(age_group, str):
                best_match, score = find_best_enum_match(age_group, AgeGroup)
                if best_match and score >= 0.8:
                    logger.info(f"Fuzzy matched age group '{age_group}' to '{best_match.value}' with score {score}")
                    validated_age_groups.append(best_match)
                else:
                    logger.warning(f"Could not match age group '{age_group}', defaulting to 'Adult'")
                    validated_age_groups.append(AgeGroup.ADULT)
            else:
                logger.warning(f"Invalid age group type, defaulting to 'Adult'")
                validated_age_groups.append(AgeGroup.ADULT)
                
        return validated_age_groups or [AgeGroup.ADULT]

    @field_validator('secondary_color_hexes')
    def validate_color_hex_length(cls, v, values):
        if len(v) != len(values.data.get('secondary_colors', [])):
            raise ValueError("Mismatch between color names and hex codes")
        for hex_code in v:
            if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", hex_code):
                raise ValueError(f"Invalid hex code format: {hex_code}")
        return v

    @field_validator('color_pairings')
    def validate_color_pairings(cls, v):
        for hex_code in v:
            if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", hex_code):
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