from dotenv import load_dotenv
import os
import json
import pandas as pd
import time
import base64
import mimetypes
import shutil
import numpy as np
from enum import Enum
from pydantic import (
    BaseModel, ValidationError, Field, 
    field_validator, model_validator
)
from typing import List, Dict, Optional, Any, Type, Set, Tuple
from anthropic import AsyncAnthropic
from anthropic._exceptions import (
    APIError, APITimeoutError, 
    APIConnectionError, RateLimitError
)
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import cohere
import logging
from datetime import datetime
from tenacity import (
    retry, stop_after_attempt, 
    wait_exponential, retry_if_exception_type, RetryCallState
)
import difflib
from pathlib import Path
import asyncio

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class APIRateConfig:
    max_retries: int = 3
    base_delay: int = 4
    max_delay: int = 60
    concurrent_limit: int = 3
    batch_size: int = 5

class APIRateLimiter:
    def __init__(self, config: APIRateConfig):
        self.semaphore = asyncio.Semaphore(config.concurrent_limit)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

    async def wait_if_needed(self):
        now = time.time()
        if now - self.last_request_time < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval)
        self.last_request_time = now

# Pricing constants for cost calculation
CLAUDE_PRICING = {
    "claude-3-5-sonnet-20240620": {
        "base_input": 3.00,        # $ per MTok
        "cache_write": 3.75,       # $ per MTok
        "cache_read": 0.30,        # $ per MTok 
        "output": 15.00            # $ per MTok
    },
    "claude-3-haiku-20240307": {
        "base_input": 0.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
        "output": 1.25
    }
}

def calculate_cost(response, model_id: str) -> dict:
    """Calculate processing cost for a single product"""
    pricing = CLAUDE_PRICING[model_id]
    
    # Input costs
    if hasattr(response.usage, 'cache_creation_input_tokens'):
        input_cost = (response.usage.cache_creation_input_tokens / 1_000_000) * pricing["cache_write"]
        cache_status = "WRITE"
    elif hasattr(response.usage, 'cache_read_input_tokens'):
        input_cost = (response.usage.cache_read_input_tokens / 1_000_000) * pricing["cache_read"]
        cache_status = "READ"
    else:  # Fallback
        input_cost = (response.usage.input_tokens / 1_000_000) * pricing["base_input"]
        cache_status = "MISS"
    
    # Output costs
    output_cost = (response.usage.output_tokens / 1_000_000) * pricing["output"]
    
    return {
        "cache_status": cache_status,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(
            f'blouse_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler()
    ]
)

# Create a separate logger for API calls
api_logger = logging.getLogger('api')
api_logger.setLevel(logging.INFO)

def log_retry_attempt(retry_state):
    """Log retry attempts with details"""
    api_logger.warning(
        f"Retry attempt {retry_state.attempt_number} "
        f"after {retry_state.outcome.exception()} "
        f"- Next retry in {retry_state.next_action.sleep} seconds"
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (APIError, APITimeoutError, APIConnectionError, RateLimitError)
    ),
    before_sleep=log_retry_attempt,
)
async def call_anthropic_api(client, **kwargs):
    """Wrapper for Anthropic API calls with retry logic"""
    try:
        return await client.messages.create(**kwargs)
    except (
        APIError, APITimeoutError, APIConnectionError, RateLimitError
    ) as e:
        api_logger.error(
            f"Anthropic API Error: {str(e)}", exc_info=True
        )
        raise
    except Exception as e:
        api_logger.error(
            f"Unexpected error in API call: {str(e)}", 
            exc_info=True
        )
        raise

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=log_retry_attempt,
)
def call_cohere_api(client, method, **kwargs):
    """Wrapper for Cohere API calls with retry logic"""
    try:
        api_method = getattr(client, method)
        return api_method(**kwargs)
    except Exception as e:
        api_logger.error(f"Cohere API Error: {str(e)}, Method: {method}", exc_info=True)
        raise

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(43)


# ---------- Enum Definitions ----------
class Color(str, Enum):
    BLACK = "Black"
    WHITE = "White"
    BROWN = "Brown"
    RED = "Red"
    YELLOW = "Yellow"
    GREEN = "Green"
    BLUE = "Blue"
    PINK = "Pink"
    ORANGE = "Orange"
    PURPLE = "Purple"
    GREY = "Grey"
    GOLD = "Gold"
    SILVER = "Silver"
    BEIGE = "Beige"
    MAROON = "Maroon"
    NAVY = "Navy"
    TEAL = "Teal"
    OLIVE = "Olive"
    CYAN = "Cyan"
    MAGENTA = "Magenta"
    OTHERS = "Others"

class ColorDetailed(str, Enum):
    # Black/White/Gray Scale
    PURE_BLACK = "Pure Black"
    SOFT_BLACK = "Soft Black"
    CHARCOAL_BLACK = "Charcoal Black"
    NIGHT_GRAY = "Night Gray"
    DEEP_GRAY = "Deep Gray"
    DIM_GRAY = "Dim Gray"
    MEDIUM_GRAY = "Medium Gray"
    DARK_GRAY = "Dark Gray"
    SILVER_GRAY = "Silver Gray"
    LIGHT_GRAY = "Light Gray"
    WHITE_SMOKE = "White Smoke"
    PURE_WHITE = "Pure White"

    # Reds/Burgundies/Wines
    CLASSIC_RED = "Classic Red"
    DEEP_PINK_RED = "Deep Pink Red"
    CRIMSON = "Crimson"
    FIRE_BRICK = "Fire Brick"
    DARK_RED = "Dark Red"
    MAROON = "Maroon"
    WINE_RED = "Wine Red"
    CARDINAL_RED = "Cardinal Red"
    DARK_PINK_RED = "Dark Pink Red"
    CRIMSON_RED = "Crimson Red"
    BURGUNDY = "Burgundy"
    DEEP_WINE = "Deep Wine"
    DARK_BURGUNDY = "Dark Burgundy"
    DEEP_BURGUNDY = "Deep Burgundy"
    DARK_WINE = "Dark Wine"

    # Pinks/Roses
    PINK = "Pink"
    LIGHT_PINK = "Light Pink"
    HOT_PINK = "Hot Pink"
    DEEP_PINK = "Deep Pink"
    PALE_VIOLET_RED = "Pale Violet Red"
    ROSE_PINK = "Rose Pink"
    DEEP_ROSE = "Deep Rose"
    MAGENTA = "Magenta"
    PASTEL_PINK = "Pastel Pink"
    BABY_PINK = "Baby Pink"
    DUSTY_ROSE = "Dusty Rose"
    MEDIUM_ROSE = "Medium Rose"

    # Blues
    CLASSIC_BLUE = "Classic Blue"
    NAVY_BLUE = "Navy Blue"
    DARK_BLUE = "Dark Blue"
    MIDNIGHT_BLUE = "Midnight Blue"
    MEDIUM_BLUE = "Medium Blue"
    ROYAL_BLUE = "Royal Blue"
    DODGER_BLUE = "Dodger Blue"
    DEEP_SKY_BLUE = "Deep Sky Blue"
    SKY_BLUE = "Sky Blue"
    POWDER_BLUE = "Powder Blue"
    LIGHT_BLUE = "Light Blue"
    LIGHT_CYAN = "Light Cyan"
    CADET_BLUE = "Cadet Blue"
    STEEL_BLUE = "Steel Blue"
    CORNFLOWER_BLUE = "Cornflower Blue"
    MEDIUM_SLATE_BLUE = "Medium Slate Blue"
    SLATE_BLUE = "Slate Blue"
    DARK_SLATE_BLUE = "Dark Slate Blue"
    OCEAN_BLUE = "Ocean Blue"
    DEEP_OCEAN_BLUE = "Deep Ocean Blue"

    # Greens
    DARK_GREEN = "Dark Green"
    FOREST_GREEN = "Forest Green"
    LIME_GREEN = "Lime Green"
    LIGHT_GREEN = "Light Green"
    PALE_GREEN = "Pale Green"
    OLIVE_GREEN = "Olive Green"
    OLIVE_DRAB = "Olive Drab"
    DARK_SEA_GREEN = "Dark Sea Green"
    SEA_GREEN = "Sea Green"
    MEDIUM_SEA_GREEN = "Medium Sea Green"
    MEDIUM_AQUAMARINE = "Medium Aquamarine"
    LIGHT_SEA_GREEN = "Light Sea Green"
    TEAL = "Teal"
    TURQUOISE = "Turquoise"
    MEDIUM_TURQUOISE = "Medium Turquoise"

    # Browns/Tans
    SADDLE_BROWN = "Saddle Brown"
    SIENNA = "Sienna"
    CHOCOLATE = "Chocolate"
    BROWN_SUGAR = "Brown Sugar"
    GOLDEN_BROWN = "Golden Brown"
    PERU = "Peru"
    BURNT_SIENNA = "Burlywood"
    OTHERS = "Others"

class Closure(str, Enum):
    BACK_HOOK = "Back Hook"
    FRONT_HOOK = "Front Hook"
    BACK_ZIPPER = "Back Zipper"
    SIDE_ZIPPER = "Side Zipper"
    FRONT_ZIP = "Front Zip"
    TIE_BACK = "Tie-Back"
    PULL_ON = "Pull-On"
    OTHERS = "Others"

class EmbellishmentLevel(str, Enum):
    NONE = "None"
    LIGHT = "Light"
    MEDIUM = "Medium"
    HEAVY = "Heavy"


class Embellishment(str, Enum):
    SEQUINS = "Sequins"
    BEADWORK = "Beadwork"
    ZARI_WORK = "Zari Work"
    HAND_EMBROIDERY = "Hand Embroidery"
    MACHINE_EMBROIDERY = "Machine Embroidery"
    LACE = "Lace"
    MIRROR_WORK = "Mirror Work"
    CUTWORK = "Cutwork"
    APPLIQUE = "Appliqué"
    PIPING = "Piping"
    BORDER_DETAIL = "Border Detail"
    TASSELS = "Tassels"
    RUFFLES = "Ruffles"
    OTHERS = "Others"

class Fit(str, Enum):
    FITTED = "Fitted"
    REGULAR = "Regular"
    RELAXED = "Relaxed"
    SLIM = "Slim"
    FORM_FITTING = "Form-Fitting"
    CROP = "Crop"
    BOXY = "Boxy"
    OTHERS = "Others"

class Hemline(str, Enum):
    STRAIGHT = "Straight"
    CROPPED = "Cropped"
    HIP_LENGTH = "Hip-length"
    NOT_SPECIFIED = "Not specified"
    OTHERS = "Others"

class Neckline(str, Enum):
    ROUND_NECK = "Round Neck"
    V_NECK = "V-Neck"
    U_NECK = "U-Neck"
    SQUARE_NECK = "Square Neck"
    SWEETHEART_NECK = "Sweetheart Neck"
    HIGH_NECK = "High Neck"
    BOAT_NECK = "Boat Neck"
    COLLAR_NECK = "Collar Neck"
    HALTER_NECK = "Halter Neck"
    OFF_SHOULDER = "Off-Shoulder"
    SCOOP_NECK = "Scoop Neck"
    OTHERS = "Others"

class Occasion(str, Enum):
    WEDDING_FUNCTIONS = "Wedding Functions"
    FESTIVE_WEAR = "Festive Wear"
    CASUAL_WEAR = "Casual Wear"
    PARTY_WEAR = "Party Wear"
    WORK_WEAR = "Work Wear"
    FORMAL_EVENTS = "Formal Events"
    BEACH_WEAR = "Beach Wear"
    BRUNCH_WEAR = "Brunch Wear"
    OTHERS = "Others"

class Pattern(str, Enum):
    SOLID = "Solid"
    FLORAL = "Floral"
    POLKA_DOT = "Polka Dot"
    STRIPED = "Striped"
    CHECKERED = "Checkered"
    ABSTRACT = "Abstract"
    ANIMAL_PRINT = "Animal Print"
    TRIBAL = "Tribal"
    EMBROIDERED = "Embroidered"
    IKAT = "Ikat"
    BROCADE = "Brocade"
    PAISLEY = "Paisley"
    OMBRE = "Ombre"
    PATCHWORK = "Patchwork"
    COLOR_BLOCKED = "Color Blocked"
    EMBELLISHED = "Embellished"
    TIE_DYE = "Tie Dye"
    WARI = "Wari"
    KALAMKARI = "Kalamkari"
    ETHNIC = "Ethnic"
    AJRAK = "Ajrak"
    GEOMETRIC = "Geometric"
    OTHERS = "Others"

class SleeveType(str, Enum):
    SLEEVELESS = "Sleeveless"
    SHORT_SLEEVES = "Short Sleeves"
    THREE_QUARTER_SLEEVES = "Three-Quarter Sleeves"
    FULL_SLEEVES = "Full Sleeves"
    ELBOW_LENGTH = "Elbow-Length"
    SPAGHETTI_STRAPS = "Spaghetti Straps"
    OFF_SHOULDER = "Off-shoulder"
    OTHERS = "Others"

class Style(str, Enum):
    TRADITIONAL = "Traditional"
    CONTEMPORARY = "Contemporary"
    FUSION = "Fusion"
    MODERN = "Modern"
    ETHNIC = "Ethnic"
    BOHEMIAN = "Bohemian"
    CASUAL = "Casual"
    FESTIVE = "Festive"
    GLAMOROUS = "Glamorous"
    OTHERS = "Others" 


class Material(str, Enum):
    SILK = "Silk"
    COTTON = "Cotton"
    CHIFFON = "Chiffon"
    GEORGETTE = "Georgette"
    BANARASI = "Banarasi"
    KANJIVARAM = "Kanjivaram"
    CHANDERI = "Chanderi"
    TUSSAR = "Tussar"
    ORGANZA = "Organza"
    LINEN = "Linen"
    MODAL_VISCOSE = "Modal_Viscose"
    OTHERS = "Others"

class Gender(str, Enum):
    WOMEN = "Women"
    MEN = "Men"
    UNISEX = "Unisex"

class AgeGroup(str, Enum):
    ADULT = "Adult"
    TEEN = "Teen"
    KIDS = "Kids"
    SENIOR = "Senior"

# ---------- Pydantic Model ----------
class BlouseAttributes(BaseModel):
    primary_color: Color
    primary_color_detailed: ColorDetailed
    primary_color_hex: str = Field(..., pattern=r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
    secondary_colors: List[Color] = Field(default_factory=list)
    secondary_color_hexes: List[str] = Field(default_factory=list)
    secondary_colors_detailed: List[ColorDetailed] = Field(default_factory=list)
    secondary_colors_detailed_hex: List[str] = Field(default_factory=list)
    pattern: List[Pattern] = Field(default_factory=list)
    sleeve_type: SleeveType
    neckline: Neckline
    closure: Closure
    fit: Fit
    hemline: Hemline
    material: Material
    # padded: bool = False
    embellishment: List[Embellishment] = Field(default_factory=list)
    embellishment_level: EmbellishmentLevel
    embellishment_detailed: List[str] = Field(default_factory=lambda: ["None"])
    occasions: List[Occasion] = Field(default_factory=list)
    occasion_detailed: List[str] = Field(default_factory=list)
    style: List[Style] = Field(default_factory=list)
    coordinating_items: dict = Field(default_factory=lambda: {
        "clothing": [],
        "accessories": [],
        "footwear": [],
        "additional_apparel": [],
        "styling_suggestions": []
    })
    gender: List[Gender] = Field(default_factory=lambda: [Gender.WOMEN])
    age_group: List[AgeGroup] = Field(default_factory=lambda: [AgeGroup.ADULT])
    color_pairings: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    
    brand: str = "suta"  
    brand_title: str
    title: str
    description: str
    price: float = Field(..., gt=0)
    care_instructions: str
    size: List[str]
    unique_design_element: List[str] = Field(
        default_factory=list,
        description="1-3 distinct selling points capturing unique aspects of the design",
        min_length=1,
        max_length=3
    )
    search_context: str = Field(
        description="Concatenated attributes for semantic search",
        default=""
    )
    text_embedding: List[float] = Field(
        default_factory=list,
        description="Cohere embedding vector of search_context"
    )


    @field_validator('color_pairings')
    def validate_color_pairings(cls, v):
        for hex_code in v:
            if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", hex_code):
                raise ValueError(f"Invalid hex code format: {hex_code}")
        return v

    @field_validator('material', mode='before')
    def validate_material(cls, v: Any) -> Material:
        """Validate material with smart compound handling."""
        if isinstance(v, Material):
            return v
        
        if not v:
            return Material.OTHERS
            
        # Handle string inputs with fuzzy matching
        if isinstance(v, str):
            # Try exact match first
            try:
                return Material(v)
            except ValueError:
                pass
                
            # Try to find best match
            best_match, score = find_best_enum_match(v, Material)
            if best_match and score >= 0.8:
                logger.info(f"Fuzzy matched material '{v}' to '{best_match.value}' with score {score}")
                return Material(best_match.value)
            
            # Handle compound materials (e.g., "Silk Cotton")
            words = v.split()
            if len(words) > 1:
                # Try each word individually
                for word in words:
                    match, score = find_best_enum_match(word, Material)
                    if match and score > 0.9:  # Higher threshold for individual words
                        logger.info(f"Matched compound material word '{word}' from '{v}' to '{match.value}'")
                        return Material(match.value)
        
        logger.warning(f"Could not match material '{v}', defaulting to 'Others'")
        return Material.OTHERS



    # @model_validator(mode='after')
    # def build_search_context(self) -> Any:
    #     # Convert lists to strings safely
    #     secondary_colors_str = ', '.join(c.value for c in self.secondary_colors) if self.secondary_colors else ''
    #     patterns_str = ', '.join(p.value for p in self.pattern) if self.pattern else ''
    #     occasions_str = ', '.join(o.value for o in self.occasions) if self.occasions else ''
    #     styles_str = ', '.join(s.value for s in self.style) if self.style else ''
    #     embellishments_str = ', '.join(e.value for e in self.embellishment) if self.embellishment else ''
        
    #     # Format coordinating items
    #     coordinating_items_parts = []
    #     if self.coordinating_items:
    #         for category, items in self.coordinating_items.items():
    #             if items:  # Only include non-empty categories
    #                 items_str = ', '.join(items)
    #                 coordinating_items_parts.append(f"{category}: {items_str}")
    #     coordinating_items_str = '; '.join(coordinating_items_parts) if coordinating_items_parts else ''
        
        # Build context parts safely
        context_parts = [
            f"Saree type: {self.saree_type.value}",
            f"Colors: {self.primary_color.value}" + (f", {secondary_colors_str}" if secondary_colors_str else ""),
            f"Materials: {self.material.value}",
            f"Patterns: {patterns_str}" if patterns_str else None,
            f"Occasions: {occasions_str}" if occasions_str else None,
            f"Styles: {styles_str}" if styles_str else None,
            f"Embellishments: {self.embellishment_level.value}" + (f" {embellishments_str}" if embellishments_str else ""),
            f"Unique features: {', '.join(self.unique_design_element)}" if self.unique_design_element else None,
            f"Coordinating items: {coordinating_items_str}" if coordinating_items_str else None
        ]

        # Filter out None values and join with periods
        context_parts = [p for p in context_parts if p is not None]
        
        # Add technical specs and description
        context_parts.extend([
            f"Technical specs: {self.length}m length, {self.weight}g weight",
            f"Details: {self.description}"
        ])
        
        self.search_context = ". ".join(context_parts).replace("  ", " ")
        return self


    @field_validator('secondary_color_hexes')
    def validate_color_hex_length(cls, v, values):
        if len(v) != len(values.data.get('secondary_colors', [])):
            raise ValueError("Mismatch between color names and hex codes")
        for hex_code in v:
            if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", hex_code):
                raise ValueError(f"Invalid hex code format: {hex_code}")
        return v

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
                logger.info(f"Fuzzy matched primary_color '{v}' to '{best_match.value}' with score {score}")
                return Color(best_match.value)
        
        logger.warning(f"Could not match primary_color '{v}', defaulting to 'Others'")
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
                logger.info(f"Fuzzy matched primary_color_detailed '{v}' to '{best_match.value}' with score {score}")
                return ColorDetailed(best_match.value)
        
        logger.warning(f"Could not match primary_color_detailed '{v}', defaulting to 'Others'")
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
                    logger.info(f"Fuzzy matched secondary color '{color}' to '{best_match.value}' with score {score}")
                    validated_colors.append(best_match)
                else:
                    logger.warning(f"Could not match secondary color '{color}', defaulting to 'Others'")
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
                    logger.info(f"Fuzzy matched secondary color '{color}' to '{best_match.value}' with score {score}")
                    validated_colors.append(best_match)
                else:
                    logger.warning(f"Could not match secondary color '{color}', defaulting to 'Others'")
                    validated_colors.append(ColorDetailed.OTHERS)
            else:
                logger.warning("Invalid secondary color type, defaulting to 'Others'")
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
                    logger.warning(f"Invalid pattern '{pattern}' → defaulting to 'Others'")
                    validated_patterns.append(Pattern.OTHERS)
            else:
                logger.warning("Invalid pattern type → defaulting to 'Others'")
                validated_patterns.append(Pattern.OTHERS)
        
        return validated_patterns or [Pattern.OTHERS]

    @field_validator('sleeve_type', mode='before')
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
                logger.info(f"Fuzzy matched sleeve_type '{v}' to '{best_match.value}' with score {score}")
                return SleeveType(best_match.value)
        
        logger.warning(f"Could not match sleeve_type '{v}', defaulting to 'Others'")
        return SleeveType.OTHERS

    @field_validator('neckline', mode='before')
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
                logger.info(f"Fuzzy matched neckline '{v}' to '{best_match.value}' with score {score}")
                return Neckline(best_match.value)
        
        logger.warning(f"Could not match neckline '{v}', defaulting to 'Others'")
        return Neckline.OTHERS

    @field_validator('closure', mode='before')
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
                logger.info(f"Fuzzy matched closure '{v}' to '{best_match.value}' with score {score}")
                return Closure(best_match.value)
        
        logger.warning(f"Could not match closure '{v}', defaulting to 'Others'")
        return Closure.OTHERS

    @field_validator('fit', mode='before')
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
                logger.info(f"Fuzzy matched fit '{v}' to '{best_match.value}' with score {score}")
                return Fit(best_match.value)
        
        logger.warning(f"Could not match fit '{v}', defaulting to 'Others'")
        return Fit.OTHERS

    @field_validator('hemline', mode='before')
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
                logger.info(f"Fuzzy matched hemline '{v}' to '{best_match.value}' with score {score}")
                return Hemline(best_match.value)
        
        logger.warning(f"Could not match hemline '{v}', defaulting to 'Others'")
        return Hemline.OTHERS

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
                logger.info(f"Fuzzy matched material '{v}' to '{best_match.value}' with score {score}")
                return Material(best_match.value)
            
            words = v.split()
            if len(words) > 1:
                for word in words:
                    match, score = find_best_enum_match(word, Material)
                    if match and score > 0.9:
                        logger.info(f"Matched compound material word '{word}' from '{v}' to '{match.value}'")
                        return Material(match.value)
        
        logger.warning(f"Could not match material '{v}', defaulting to 'Others'")
        return Material.OTHERS

    @field_validator('embellishment', mode='before')
    def validate_embellishment(cls, v: Any) -> List[Embellishment]:
        if not v:
            return []
            
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
                    logger.warning(f"Could not match embellishment '{embellishment}', defaulting to 'Others'")
                    validated_embellishments.append(Embellishment.OTHERS)
            else:
                logger.warning("Invalid embellishment type, defaulting to 'Others'")
                validated_embellishments.append(Embellishment.OTHERS)
                
        return validated_embellishments

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
                logger.info(f"Fuzzy matched embellishment_level '{v}' to '{best_match.value}' with score {score}")
                return best_match
        
        logger.warning(f"Could not match embellishment_level '{v}', defaulting to 'None'")
        return EmbellishmentLevel.NONE

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
                logger.warning("Invalid occasion type, defaulting to 'Others'")
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
                logger.warning("Invalid style type, defaulting to 'Others'")
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
                logger.warning("Invalid gender type, defaulting to 'Women'")
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
                    logger.info(f"Fuzzy matched age_group '{age_group}' to '{best_match.value}' with score {score}")
                    validated_age_groups.append(best_match)
                else:
                    logger.warning(f"Could not match age_group '{age_group}', defaulting to 'Adult'")
                    validated_age_groups.append(AgeGroup.ADULT)
            else:
                logger.warning("Invalid age_group type, defaulting to 'Adult'")
                validated_age_groups.append(AgeGroup.ADULT)
                
        return validated_age_groups or [AgeGroup.ADULT]

    @field_validator('secondary_colors_detailed_hex')
    def validate_secondary_colors_detailed_hex(cls, v):
        if not isinstance(v, list):
            raise ValueError("Must be a list of hex codes")
        for hex_code in v:
            if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", hex_code):
                raise ValueError(f"Invalid hex code format: {hex_code}")
        return v
    
    @model_validator(mode='after')
    def validate_enums(self):
        errors = []
        
        # Special handling for pattern field
        if hasattr(self, 'pattern'):
            validated_patterns = []
            for pattern in self.pattern:
                if pattern not in Pattern._value2member_map_:
                    logger.warning(f"Invalid pattern value '{pattern}' found, defaulting to 'Others'")
                    validated_patterns.append(Pattern.OTHERS)
                else:
                    validated_patterns.append(pattern)
            self.pattern = validated_patterns if validated_patterns else [Pattern.OTHERS]
        
        # Validate all other enum fields
        for field_name, field_info in self.model_fields.items():
            if field_name == 'pattern':
                continue  # Skip pattern as it's handled above
                
            field_value = getattr(self, field_name)
            field_type = field_info.annotation
            
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                if isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        if item not in field_type._value2member_map_:
                            errors.append(f"{field_name}[{i}]: '{item}' not in {field_type.__name__}")
                else:
                    if field_value not in field_type._value2member_map_:
                        errors.append(f"{field_name}: '{field_value}' not in {field_type.__name__}")
        
        if errors:
            self.validation_errors = errors
            raise ValidationError(errors)
            
        return self


    @model_validator(mode='after')
    def build_search_context(self) -> Any:
        secondary_colors_str = ', '.join(c.value for c in self.secondary_colors) if self.secondary_colors else ''
        patterns_str = ', '.join(p.value for p in self.pattern) if self.pattern else ''
        occasions_str = ', '.join(self.occasion_detailed) if self.occasion_detailed else ''
        styles_str = ', '.join(s.value for s in self.style) if self.style else ''
        embellishments_str = ', '.join(self.embellishment_detailed) if self.embellishment_detailed else ''
        
        context_parts = [
            f"Colors: {self.primary_color.value}" + (f", {secondary_colors_str}" if secondary_colors_str else ""),
            f"Materials: {self.material.value}",
            f"Patterns: {patterns_str}" if patterns_str else None,
            f"Sleeve Type: {self.sleeve_type.value}",
            f"Neckline: {self.neckline.value}",
            f"Closure: {self.closure.value}",
            f"Fit: {self.fit.value}",
            # f"Padded: {self.padded}",
            f"Hemline: {self.hemline.value}",
            f"Occasions: {occasions_str}" if occasions_str else None,
            f"Styles: {styles_str}" if styles_str else None,
            f"Embellishments: {self.embellishment_level.value}" + (f" {embellishments_str}" if embellishments_str else ""),
            f"Unique features: {', '.join(self.unique_design_element)}" if self.unique_design_element else None,
        ]
        
        context_parts = [p for p in context_parts if p is not None]
        context = '. '.join(context_parts)
        
        context_parts.extend([
            f"Details: {self.description}"
        ])
        
        self.search_context = context
        return self

# === Precomputed Enum Mappings ===
ENUM_MAPPINGS = {
    enum_cls.__name__: {item.name: item.value for item in enum_cls}
    for enum_cls in [
        Color, ColorDetailed, Pattern, Material, Closure, Fit, Hemline, Neckline, 
        Occasion, Pattern, SleeveType, Style, Material, Gender, AgeGroup
    ]
}

# === Centralized Image Handling ===
@lru_cache(maxsize=100)
def _read_and_encode_image(path: str) -> tuple:
    """Returns (base64_str, mime_type) for both APIs"""
    with open(path, "rb") as f:
        data = f.read()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return base64.b64encode(data).decode("utf-8"), mime_type

def image_to_data_url(image_path: str) -> str:
    """Convert image to data URL format required by Cohere"""
    _, file_extension = os.path.splitext(image_path)
    file_type = file_extension[1:]  # Remove the dot
    base64_str, _ = _read_and_encode_image(image_path)
    return f"data:image/{file_type};base64,{base64_str}"

def create_image_message(image_path: str) -> dict:
    base64_str, mime_type = _read_and_encode_image(image_path)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": base64_str}
    }


# === Image Copy Pooling ===
class ImageCopier:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}

    def copy_image_async(self, src_path: str, dest_folder: str) -> None:
        future = self.executor.submit(shutil.copy2, src_path, dest_folder)
        self.futures[future] = (src_path, dest_folder)

    def get_completed(self):
        for future in as_completed(self.futures):
            src, dest = self.futures[future]
            try:
                yield future.result()
            except Exception as e:
                print(f"Error copying {src} to {dest}: {str(e)}")

# === Incremental Saving ===
class IncrementalSaver:
    def __init__(self, json_path: str, excel_path: str):
        self.json_path = json_path
        self.excel_path = excel_path
        self.buffer = []
        self.buffer_size = 3  # Reduced from 10 to match batch size
        self.processed_products = self._load_processed_products()
        
    def _load_processed_products(self) -> set:
        processed = set()
        
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return processed
                        
                    # Split content into individual JSON objects
                    json_objects = []
                    current_obj = ""
                    brace_count = 0
                    
                    for char in content:
                        current_obj += char
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            
                        if brace_count == 0 and current_obj.strip():
                            try:
                                data = json.loads(current_obj)
                                # Each object is {product_id: {...details...}}
                                processed.update(data.keys())
                                current_obj = ""
                            except json.JSONDecodeError:
                                continue
                            
            except Exception as e:
                logger.error(f"Error reading JSON file: {str(e)}")
        
        if processed:
            logger.info(f"Loaded {len(processed)} processed products")
        return processed

    def save(self, product_id: str, result: dict):
        if product_id in self.processed_products:
            return
            
        # Convert Timestamp to string for JSON serialization
        if 'processed_at' in result:
            result['processed_at'] = str(result['processed_at'])
            
        self.buffer.append((product_id, result))
        self.processed_products.add(product_id)
        
        # Always flush after adding to ensure immediate save
        self._flush_buffer()  # Changed from conditional flush
    
    def _flush_buffer(self):
        if not self.buffer:
            return
            
        # Read existing content
        existing_data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Split content into individual JSON objects
                        current_obj = ""
                        brace_count = 0
                        
                        for char in content:
                            current_obj += char
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                
                            if brace_count == 0 and current_obj.strip():
                                try:
                                    data = json.loads(current_obj)
                                    existing_data.update(data)
                                    current_obj = ""
                                except json.JSONDecodeError:
                                    continue
                                    
            except Exception as e:
                logger.error(f"Error reading existing JSON data: {str(e)}")
        
        # Update with new data
        for product_id, result in self.buffer:
            ordered_result = self._prepare_ordered_result(product_id, result)
            existing_data[product_id] = ordered_result
            
        # Write all data back
        with open(self.json_path, 'w') as f:
            for i, (product_id, data) in enumerate(existing_data.items()):
                if i > 0:
                    f.write('\n')
                json.dump({product_id: data}, f)
                
        self.buffer = []

    def _prepare_ordered_result(self, product_id: str, result: dict) -> dict:
        ordered_result = {}
        
        # Basic product info
        ordered_result['brand'] = result.get('brand', '')
        ordered_result['brand_title'] = result.get('brand_title', '')
        ordered_result['title'] = result.get('title', '')
        ordered_result['description'] = result.get('description', '')
        ordered_result['price'] = result.get('price', 0)

        # Core attributes
        ordered_result['material'] = result.get('material', '')
        ordered_result['fit'] = result.get('fit', '')
        ordered_result['hemline'] = result.get('hemline', '')
        ordered_result['neckline'] = result.get('neckline', '')
        ordered_result['sleeve_type'] = result.get('sleeve_type', '')
        ordered_result['closure'] = result.get('closure', '')
        # ordered_result['padded'] = result.get('padded', False)

        # Color information
        ordered_result['primary_color'] = result.get('primary_color', '')
        ordered_result['primary_color_detailed'] = result.get('primary_color_detailed', '')
        ordered_result['primary_color_hex'] = result.get('primary_color_hex', '')
        ordered_result['secondary_colors'] = result.get('secondary_colors', [])
        ordered_result['secondary_colors_detailed'] = result.get('secondary_colors_detailed', [])
        ordered_result['secondary_color_hexes'] = result.get('secondary_color_hexes', [])
        ordered_result['secondary_colors_detailed_hex'] = result.get('secondary_colors_detailed_hex', [])
        ordered_result['color_pairings'] = result.get('color_pairings', [])
        
        
        # Embellishments
        ordered_result['embellishment_level'] = result.get('embellishment_level', '')
        ordered_result['embellishment'] = result.get('embellishment', [])
        ordered_result['embellishment_detailed'] = result.get('embellishment_detailed', [])
        
        # Style and occasions
        ordered_result['style'] = result.get('style', [])
        ordered_result['occasions'] = result.get('occasions', [])
        ordered_result['occasion_detailed'] = result.get('occasion_detailed', [])
          
        # Additional information
        ordered_result['size'] = result.get('size', [])
        ordered_result['care_instructions'] = result.get('care_instructions', '')
        ordered_result['unique_design_element'] = result.get('unique_design_element', [])
        ordered_result['coordinating_items'] = result.get('coordinating_items', {})
        
        # Target audience
        ordered_result['gender'] = result.get('gender', [])
        ordered_result['age_group'] = result.get('age_group', [])
        
        # URLs and embeddings
        ordered_result['image_urls'] = result.get('image_urls', [])
        ordered_result['product_url'] = result.get('product_url', '')
        ordered_result['image_embedding'] = result.get('image_embedding', [])
        ordered_result['text_embedding'] = result.get('text_embedding', [])
        ordered_result['search_context'] = result.get('search_context', '')
        
        # Processing metadata
        ordered_result['processed_at'] = result.get('processed_at', '')
        ordered_result['inference_time'] = result.get('inference_time', 0)
        ordered_result['input_tokens'] = result.get('input_tokens', 0)
        ordered_result['output_tokens'] = result.get('output_tokens', 0)
        ordered_result['product_type'] = 'blouse'
        return ordered_result

    def _prepare_row(self, result: dict) -> dict:
        row = {'product_id': result['product_id']}
        
        # Extract only the product attributes
        if result:
            # Get the main attributes from the result
            attributes = {
                'primary_color', 'primary_color_detailed', 'primary_color_hex',
                'secondary_colors', 'secondary_color_hexes', 'secondary_colors_detailed',
                'pattern', 'material', 'neckline', 'sleeve_type', 'closure',
                'fit', 'hemline', 'length', 'width', 'weight',
                'occasions', 'occasion_detailed', 'embellishment_level',
                'embellishment', 'embellishment_detailed', 'style',
                'brand', 'brand_title', 'title', 'description', 'price',
                'care_instructions', 'size', 'unique_design_element',
                'search_context'
            }
            
            for key, value in result.items():
                if key in attributes:
                    if isinstance(value, (list, tuple)):
                        row[key] = ', '.join(map(str, value))
                    else:
                        row[key] = value
                        
            # Handle coordinating items separately to make it cleaner
            if 'coordinating_items' in result:
                coord_items = result['coordinating_items']
                for key in ['clothing', 'accessories', 'footwear', 'additional_apparel', 'styling_suggestions']:
                    if key in coord_items:
                        row[f'coordinating_{key}'] = ', '.join(map(str, coord_items[key]))
                        
        return row

# === Batch Processing ===
class ProcessingStats:
    def __init__(self):
        self.successful_products = {}  # product_id -> processing time
        self.failed_products = {}  # product_id -> error reason
        self.total_tokens = 0
        self.total_time = 0
        
    def add_success(self, product_id: str, processing_time: float, tokens: int):
        self.successful_products[product_id] = processing_time
        self.total_tokens += tokens
        self.total_time += processing_time
        logger.info(f"✓ Successfully processed product {product_id} in {processing_time:.2f}s (Tokens: {tokens})")
        
    def add_failure(self, product_id: str, error_reason: str):
        self.failed_products[product_id] = error_reason
        logger.error(f"✗ Failed to process product {product_id}: {error_reason}")
        
    def print_summary(self):
        total_products = len(self.successful_products) + len(self.failed_products)
        success_rate = (len(self.successful_products)/total_products)*100 if total_products > 0 else 0
        avg_time = self.total_time/max(1, len(self.successful_products))
        
        summary = [
            "\nProcessing Summary:",
            "=" * 50,
            f"\nSuccessfully Processed Products ({len(self.successful_products)}):",
            *[f"✓ Product {pid}: {time:.2f}s" for pid, time in self.successful_products.items()],
            f"\nFailed Products ({len(self.failed_products)}):",
            *[f"✗ Product {pid}: {reason}" for pid, reason in self.failed_products.items()],
            "\nOverall Statistics:",
            f"Total Products Attempted: {total_products}",
            f"Success Rate: {success_rate:.1f}%",
            f"Total Processing Time: {self.total_time:.1f}s",
            f"Average Time Per Success: {avg_time:.1f}s",
            f"Total Tokens Used: {self.total_tokens}",
            "=" * 50
        ]
        
        summary_text = "\n".join(summary)
        logger.info(summary_text)
        print(summary_text)

async def batch_process_products(
    df: pd.DataFrame, 
    saver: IncrementalSaver, 
    batch_size: int = 10,
    rate_limiter: Optional[APIRateLimiter] = None
) -> dict:
    results = {}
    stats = ProcessingStats()
    total_products = len(df)
    total_batches = (total_products + batch_size - 1) // batch_size

    rows = df.to_dict('records')
    
    logger.info(f"Starting processing of {total_products} products in {total_batches} batches")
    logger.info("Each batch takes about 30-45 seconds to process\n")
    
    for batch_num in range(0, total_products, batch_size):
        start_time = time.time()
        batch = rows[batch_num:batch_num + batch_size]
        
        # Process batch concurrently
        tasks = [process_product(row, rate_limiter) for row in batch]
        batch_results = await asyncio.gather(*tasks)
        
        for product_id, result in batch_results:
            if result:
                results[product_id] = result
                processing_time = result['inference_time']
                tokens = result['input_tokens'] + result['output_tokens']
                stats.add_success(product_id, processing_time, tokens)
                saver.save(product_id, result)
            else:
                stats.add_failure(product_id, "No result returned")
        
        # Dynamic delay based on batch processing time
        batch_time = time.time() - start_time
        if batch_num < total_products - batch_size:
            # If batch took less than 1 second, wait 1 second
            # If batch took more than 1 second, wait 0.5 seconds
            delay = 1 if batch_time < 1 else 0.5
            await asyncio.sleep(delay)
        
        logger.info(f"Batch {batch_num // batch_size + 1} completed in {time.time() - start_time:.1f}s")
    
    stats.print_summary()
    return results


# === Updated Main Function ===
async def main():
    global ANTHROPIC_CLIENT, COHERE_CLIENT
    
    load_dotenv()
    # Initialize both clients
    ANTHROPIC_CLIENT = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    COHERE_CLIENT = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    
    # Initialize rate limiter
    config = APIRateConfig()
    rate_limiter = APIRateLimiter(config)
    
    # Initialize components
    saver = IncrementalSaver(
        json_path='/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/prodcut_desc_claude/blouse_attributes/blouse_attributes_results.json',
        excel_path='/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/prodcut_desc_claude/blouse_attributes/blouse_attributes_results.xlsx'
    )
    
    # Create output directories
    os.makedirs('/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/prodcut_desc_claude/blouse_attributes', exist_ok=True)
    os.makedirs('/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/prodcut_desc_claude/blouse_attributes/images', exist_ok=True)
    
    try:
        print("\nStarting blouse attribute extraction...")
        
        # Load data efficiently
        df = pd.read_csv(
            '/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/uplyft-shopify-scraper/data/02_intermediate/processed_dataframe.csv'
        )
        df = df[
            df['Product Type'].str.contains('Blouse|garage_blouse', case=False, na=False)
        ].copy()

        # Limit to 12 products for testing
        # df = df.head(10)
        
        # Filter out already processed products
        df = df[~df['Product ID'].astype(str).isin(saver.processed_products)]
        
        print(f"\nFound {len(df)} new blouse products to process")

        if len(df) == 0:
            print("No new blouse products found to process")
            return pd.DataFrame()
            
        # Add image paths
        df['image_path'] = df.apply(
            lambda row: os.path.join(
                "/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/uplyft-shopify-scraper/data/02_intermediate/suta/images/",
                str(row['Brand Name']),
                f"{row['Product ID']}.jpg"
            ),
            axis=1
        )
        
        # Filter out rows with missing images
        df = df[df['image_path'].apply(os.path.exists)]
        
        if len(df) == 0:
            print("No valid products found with images")
            return pd.DataFrame()
            
        print(f"\nProcessing {len(df)} products with valid images...")
        
        # Process in optimized batches
        batch_start_time = time.time()
        results = await batch_process_products(df, saver, rate_limiter=rate_limiter)
        batch_time = time.time() - batch_start_time
        
        if not results:
            print("No results generated")
            return
        
        # Final flush of any remaining items
        saver._flush_buffer()
        
        total_time = time.time() - batch_start_time
        
        # Calculate statistics
        successful_products = len([r for r in results.values() if r])
        total_tokens = sum(
            r.get('input_tokens', 0) + r.get('output_tokens', 0) 
            for r in results.values() if r
        )
        avg_inference_time = sum(
            r.get('inference_time', 0) 
            for r in results.values() if r
        ) / successful_products if successful_products else 0
        
        print("\nProcessing Summary:")
        print(f"Total products attempted: {len(df)}")
        print(f"Successfully processed: {successful_products}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Average inference time per product: {avg_inference_time:.2f}s")
        print(f"\nTiming Breakdown:")
        print(f"Batch processing time: {batch_time:.2f}s")
        print(f"Total execution time: {total_time:.2f}s")
        print("\nResults saved to:")
        print(f"- JSON: {saver.json_path}")
        print(f"- Excel: {saver.excel_path}")
        
        # Convert results to DataFrame
        result_rows = []
        for product_id, result in results.items():
            if result and 'data' in result:
                row = {'product_id': product_id}
                row.update(result['data'])
                row.update({
                    'processed_at': result.get('processed_at'),
                    'inference_time': result.get('inference_time'),
                    'input_tokens': result.get('input_tokens'),
                    'output_tokens': result.get('output_tokens')
                })
                result_rows.append(row)
        
        return pd.DataFrame(result_rows)
        
    except Exception as e:
        logger.error("Error in main execution", exc_info=True)
        raise

# === Enum Optimization in Split Attributes ===
@lru_cache(maxsize=None)
def convert_enum(value, enum_name):
    return ENUM_MAPPINGS[enum_name].get(value, value)

def split_attributes(data: dict) -> dict:
    # Optimized enum conversion using precomputed mappings
    enum_conversion_map = {
        'primary_color': 'Color',
        'primary_color_detailed': 'ColorDetailed', 
        'secondary_colors': 'Color',
        'secondary_colors_detailed': 'ColorDetailed',
        'pattern': 'Pattern',
        'sleeve_type': 'SleeveType',
        'neckline': 'Neckline',
        'closure': 'Closure',
        'fit': 'Fit',
        'hemline': 'Hemline',
        'material': 'Material',
        'embellishment': 'Embellishment',
        'embellishment_level': 'EmbellishmentLevel',
        'occasions': 'Occasion',
        'style': 'Style',
        'gender': 'Gender',
        'age_group': 'AgeGroup'
    }
    result = {}
    
    # Convert enum values
    for field, enum_name in enum_conversion_map.items():
        if field in data:
            value = data[field]
            if isinstance(value, list):
                result[field] = [convert_enum(v, enum_name) for v in value]
            else:
                result[field] = convert_enum(value, enum_name)
    
    # Copy non-enum fields
    non_enum_fields = [
        'primary_color_hex', 'secondary_color_hexes',
        'secondary_colors_detailed_hex', 'embellishment_detailed',
        'coordinating_items', 'color_pairings', 'brand', 'brand_title',
        'title', 'description', 'price', 'care_instructions',
        'size', 'unique_design_element', 'search_context',
        'text_embedding', 'image_embedding', 'image_urls', 'product_url'
    ]
    
    for field in non_enum_fields:
        if field in data:
            result[field] = data[field]
    
    return result

# === Enum Validation Helpers ===
def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing special chars and converting to lowercase."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def get_word_variations(word: str) -> Set[str]:
    """Generate common variations of a word."""
    word = word.lower()
    variations = {word}
    
    # Common suffixes and their base forms
    suffix_map = {
        'ed': '', 'ing': '', 'tion': 't', 
        'sion': 's', 'al': '', 'ic': '',
        'y': '', 'ies': 'y', 'ied': 'y'
    }
    
    for suffix, replacement in suffix_map.items():
        if word.endswith(suffix):
            variations.add(word[:-len(suffix)] + replacement)
    
    return variations

def build_enum_variations(enum_class: Type[Enum]) -> Dict[str, Enum]:
    """Build a mapping of normalized variations to enum values."""
    variations = {}
    
    for enum_value in enum_class:
        # Get base value and its variations
        base_value = enum_value.value
        words = normalize_text(base_value).split()
        
        # Add original value
        variations[normalize_text(base_value)] = enum_value
        
        # Add variations for each word
        for word in words:
            word_vars = get_word_variations(word)
            for var in word_vars:
                variations[var] = enum_value
                
        # Handle compound values (e.g., "Cotton Silk" -> ["Cotton", "Silk"])
        if len(words) > 1:
            for word in words:
                variations[normalize_text(word)] = enum_value
    
    return variations

def get_similarity_score(text1: str, text2: str) -> float:
    """Get similarity score between two texts."""
    return difflib.SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def find_best_enum_match(value: str, enum_class: Type[Enum], threshold: float = 0.8) -> Tuple[Optional[Enum], float]:
    """Find the best matching enum value using multiple strategies."""
    if not value:
        return None, 0.0
        
    normalized_value = normalize_text(value)
    
    # Try exact matches first
    variations = build_enum_variations(enum_class)
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

def validate_enum_value(
    value: Any, 
    enum_class: Type[Enum], 
    field_name: str, 
    allow_none: bool = False
) -> Optional[Enum]:
    """Generic enum validation with smart matching."""
    if value is None:
        if allow_none:
            return None
        return next((enum_class[name] for name in ['OTHER', 'OTHERS', 'NONE'] 
                    if name in enum_class.__members__), next(iter(enum_class)))
    
    if isinstance(value, enum_class):
        return value
        
    if isinstance(value, str):
        match, score = find_best_enum_match(value, enum_class)
        if match and score > 0.8:
            if score < 1.0:
                logger.info(f"Using fuzzy match '{match.value}' (score: {score:.2f}) for {field_name} value '{value}'")
            return match
            
    # Try to find default value
    try:
        return enum_class('OTHERS')
    except ValueError:
        try:
            return enum_class('OTHER')
        except ValueError:
            return next(iter(enum_class))

def validate_enum_list(values: List[Any], enum_class: Type[Enum], field_name: str) -> List[Enum]:
    """Validate a list of enum values."""
    if not isinstance(values, list):
        values = [values]
        
    result = []
    for value in values:
        enum_value = validate_enum_value(value, enum_class, field_name)
        if enum_value:
            result.append(enum_value)
            
    return result if result else [validate_enum_value(None, enum_class, field_name)]

# ---------- Core Functions ---------- 
def get_product_attributes(product_type: str) -> dict:
    attribute_instructions = {
        "inferred_attributes": {
            k: f"{v} (Allowed values: {', '.join(e.__members__.keys())})" 
            for k, v, e in [
                ("primary_color", "Dominant color", Color),
                ("primary_color_detailed", "Detailed color description", ColorDetailed),
                ("secondary_colors", "Secondary colors", Color),
                ("secondary_colors_detailed", "Detailed secondary colors", ColorDetailed),
                ("pattern", "Main pattern", Pattern),
                ("material", "Fabric material", Material),
                ("neckline", "Neckline style", Neckline),
                ("sleeve_type", "Type of sleeves", SleeveType),
                ("closure", "Closure type", Closure),
                ("fit", "Fit type", Fit),
                ("hemline", "Hemline style", Hemline),
                ("embellishment_level", "Level of embellishment", EmbellishmentLevel),
                ("embellishment", "Embellishment type", Embellishment),
                ("style", "Overall style", Style),
                ("gender", "Target gender", Gender),
                ("age_group", "Target age group", AgeGroup)
            ]
        },
        "given_attributes": {
            "brand": "Exact brand name",
            "brand_title": "Brand collection/line name", 
            "title": "Product title",
            "description": "Product description",
            "price": "Price in INR",
            "care_instructions": "Care instructions",
            "size": "Available sizes"
        }
    }
    

    return attribute_instructions

def log_errors(errors):
    """Log validation errors to console."""
    for error in errors:
        print(f"Validation error: {error}")

def save_partial_data(data):
    """Save partially valid data to console."""
    print("Partial data saved:", json.dumps(data, indent=2))

def log_complete_failure(product_id):
    """Log complete processing failures."""
    print(f"Complete failure processing product {product_id}")

# Add client as module-level constant after main() definition
ANTHROPIC_CLIENT = None

# ===== CACHEABLE SYSTEM PROMPT =====
SYSTEM_PROMPT = f"""You are a fashion expert specializing in Indian ethnic wear, particularly blouses for sarees. 
        Analyze this blouse image and product information to generate detailed JSON data.
        
        IMPORTANT GUIDELINES:
        - Focus ONLY on the blouse in the image, not any saree or other garments shown
        - Use the exact product name from the data for brand_title (e.g. if product name is "Green Dream", use that)
        - Analyze colors ONLY from the blouse itself, not from any accompanying saree or accessories
        - Be precise in identifying embellishments and patterns that are actually present on the blouse
        - STRICTLY use only the allowed values provided for each attribute
        - Use material information from the product data, don't try to infer it from the image
        - For ruffled designs, use "Others" as pattern and "Sequinned" as embellishment
        - Color pairings MUST be 6-digit hex codes (e.g., "#FF0000", "#00FF00")
        - Use the exact product URL from the data for product_url field
        
        Requirements:
        1. Colors & Patterns:
           - "primary_color": Identify and list the single most dominant color in the product as primary_color. primary_color MUST be one of: {', '.join(Color.__members__.keys())} (ONLY for the blouse). 
           - "primary_color_hex": Provide the exact hex code for the primary color (e.g., #FF5733)
           - "primary_color_detailed": Identify and list the single most dominant color in the product as primary_color_detailed. primary_color_detailed MUST be one of: ColorDetailed (ONLY for the blouse)
           - "secondary_colors": Identify and list the next most prominent colors in the product, secondary_colors MUST be one of: {', '.join(Color.__members__.keys())} and ONLY for the blouse
           - "secondary_color_hexes": Provide the exact hex code for each secondary color in a comma-separated python list format (e.g., #33FF57, #5733FF)
           - "secondary_colors_detailed": Identify and list the next most prominent colors from ONLY the blouse, if any in a comma-separated python list format. For each secondary color, provide a closest matching secondary_colors_detailed from ColorDetailed enum
           - "secondary_colors_detailed_hex": REQUIRED - Provide the exact hex code for each secondary_colors_detailed in a comma-separated python list format. Must match the order of secondary_colors_detailed.
           - "color_pairings": List of complementary color hex codes that can be used to style the product in a comma-separated python list format. MUST be a list of EXACTLY 5 hex codes (e.g., ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"])
           - "pattern": Describe the patterns only in BLOUSE in a comma-separated python list format. pattern MUST ONLY use these values: {', '.join(Pattern.__members__.keys())} (based on patterns in blouse ONLY). Make sure to include only patterns and not embellishments that are actually present in the blouse.
           - If primary_color or secondary_colors are not found in the {', '.join(Color.__members__.keys())}, use "Others"
           - If pattern doesn't match exactly, use 'Others'
           - All color hex codes must be valid 6-digit formats (#RRGGBB)
        
        2. Design Elements:
           - "sleeve_type": Must be one of: {', '.join(SleeveType.__members__.keys())}
           - "neckline": Must be one of: {', '.join(Neckline.__members__.keys())}
           - "closure": Must be one of: {', '.join(Closure.__members__.keys())}
           - "fit": Must be one of: {', '.join(Fit.__members__.keys())}
           - "hemline": Must be one of: {', '.join(Hemline.__members__.keys())}
    
        
        3. Material & Structure:
           - "material": MUST be one of: {', '.join(Material.__members__.keys())}. MUST use FIRST valid Material enum value found in brand's product description. Example: "Silk Cotton with Zari" → "Silk". If no exact matches in {', '.join(Material.__members__.keys())} found → 'Other'

           
        4. Usage & Style:
           - "occasions": select 2-4 occasions and they MUST ONLY use these values: {', '.join(Occasion.__members__.keys())}
           - "occasions_detailed": List 3-5 high-impact events for maximum visibility. Include Celebrity weddings/fashion weeks, Cultural festivals (Diwali, Onam, Durga Puja), Fusion events (pride parades, date nights, gallery openings), Royal gatherings/red carpet events. Format: Python list
           - "embellishment_level": embellishment_level MUST be one of: {', '.join(EmbellishmentLevel.__members__.keys())}
           - "embellishment": embellishment MUST ONLY use these values: {', '.join(Embellishment.__members__.keys())}
           - If embellishment doesn't match exactly, use the closest match or mention "None"
           - embellishment_detailed: embellishment_detailed MUST be a list of 3-5 detailed embellishments present in the blouse and only if the blouse has any embellishments. Detail any decorative elements or embellishments on the product in a comma-separated python list format for embellishment_detailed
           - "style": style MUST be one of: {', '.join(Style.__members__.keys())}. Characterize the overall style of the product using 1-2 descriptors into a list format
           
        5. Product Information:
           - "brand_title": Use the exact product name from column "Product Name" for brand_title
           - "title": A memorable and creative product title (30-60 characters) incorporating Indian culture, wordplay, and brand style (eg: 1. Bloom Bazaar, 2. Petal Pirouette, 3. Floral Rhapsody, 4. Blossom Breeze, 5. Botanical Bliss)
           - "description": A detailed product description (100-150 words) written from an expert perspective, emphasizing unique features, benefits, and selling points with the quirky title we generated now.
           
        6. Coordinating Items:
           - "clothing": REQUIRED - Suggest EXACTLY 5 coordinate clothing items that would pair well with the main product. Consider:
             * Traditional options (e.g., blouses, cholis)
             * Contemporary options (e.g., crop tops, jackets)
             * Color coordination with the blouse's primary and secondary colors
             * Style matching (traditional with traditional, modern with modern)
             * Occasion appropriateness
           - "accessories": REQUIRED - Suggest EXACTLY 5 accessories that enhance the look. Consider:
             * Jewelry (necklaces, earrings, bangles)
             * Hair accessories (pins, clips, tikkas)
             * Bags (clutches, potlis)
             * Color matching or complementary colors
             * Style consistency
           - "footwear": REQUIRED - Recommend EXACTLY 5 footwear options. Consider:
             * Traditional options (juttis, kolhapuris)
             * Modern options (heels, flats)
             * Comfort and occasion appropriateness
             * Color coordination
             * Style matching
           - "additional_apparel": REQUIRED - Suggest EXACTLY 3 additional items. Consider:
             * Layering pieces (shawls, jackets)
             * Weather-appropriate options
             * Style enhancement pieces
             * Color complementing items
           - "styling_suggestions": REQUIRED - Provide EXACTLY 3 specific outfit combinations using items from above categories. Each suggestion should:
             * Specify items from at least 3 categories
             * Include the occasion or setting
             * Consider the overall style theme
             * Ensure color coordination
             * Maintain style consistency
           
        7. Target Audience:
           - "gender": MUST be from: {', '.join(Gender.__members__.keys())}
           - "age_group": MUST be from: {', '.join(AgeGroup.__members__.keys())}
           
        8. Unique Selling Points:
           - unique_design_element: List 1-3 distinct selling points that make this item unique. Focus on:
             * Eye-catching design elements (e.g., "Signature scalloped border with gold threadwork")
             * Innovative fabric technology (e.g., "Wrinkle-resistant silk for easy maintenance")
             * Versatile styling (e.g., "Reversible design offers two looks in one saree")
             * Fashion trends (e.g., "On-trend ombré effect fading from navy to silver")
             * Craftsmanship (e.g., "Hand-embroidered peacock motif takes 72 hours to complete")
           
        VALIDATION RULES:
        - All enum values must match exactly with the provided options
        - No custom values or variations are allowed
        - For occasions, use "Bridal/Wedding" instead of "Wedding" or "Wedding Wear"
        - For materials, use only the standard options provided (e.g., use "Silk" for artificial silk)
        - For patterns, use "Others" if the exact pattern is not in the list
        - For embellishments, choose the closest match from allowed values
        - All color hex codes must be 6 digits (e.g., "#FF0000" not "#F00")
        - color_pairings must be a list of EXACTLY 5 hex codes
        - coordinating_items MUST NOT contain empty arrays - each category must have the minimum number of suggestions specified
           
        Follow the schema exactly:
        <schema>{BlouseAttributes.schema_json(indent=2)}</schema>
        
        Return only valid JSON wrapped in <json></json> tags."""

async def process_product(row: dict, rate_limiter: Optional[APIRateLimiter] = None) -> tuple:
    """Process a single product with optional rate limiting"""
    if rate_limiter:
        async with rate_limiter.semaphore:
            await rate_limiter.wait_if_needed()
            return await process_product_internal(row)
    return await process_product_internal(row)

async def process_product_internal(row: dict) -> tuple:
    try:
        product_id = str(row['Product ID'])
        image_path = row['image_path']
        
        logger.info(f"Starting processing for product {product_id}")
        
        # Get image URLs and product URL from the row
        image_urls = eval(row['Product Image Link']) if isinstance(row['Product Image Link'], str) else []
        product_url = row['Product URL'] if pd.notna(row['Product URL']) else ""
        
        # Copy image to output directory
        output_image_path = os.path.join(
            '/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/prodcut_desc_claude/blouse_attributes/images',
            f"{product_id}.jpg"
        )
        try:
            shutil.copy2(image_path, output_image_path)
            logger.info(f"✓ Copied image for product {product_id}")
        except Exception as e:
            logger.error(f"✗ Failed to copy image for product {product_id}: {str(e)}", exc_info=True)
            return product_id, None
        
        logger.info(f"Generating image embedding for product {product_id}")
        # Generate image embedding using Cohere
        try:
            data_url = image_to_data_url(image_path)
            image_embedding = call_cohere_api(
                COHERE_CLIENT,
                'embed',
                model="embed-english-v3.0",
                texts=[],
                images=[data_url],
                input_type="image",
                embedding_types=["float"]
            ).embeddings.float[0]
            logger.info(f"Successfully generated image embedding for product {product_id}")
        except Exception as e:
            logger.error(f"Failed to generate image embedding for product {product_id}: {str(e)}", exc_info=True)
            return product_id, None

        messages = [
            {
                "role": "user",
                "content": [
                    create_image_message(image_path),
                    {
                        "type": "text",
                        "text": f"Product Info: {row}"
                        # "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]

        start_time = time.time()
        
        try:
            response = await ANTHROPIC_CLIENT.messages.create(
                model="claude-3-5-sonnet-20240620",
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=messages,
                max_tokens=4096,
                temperature=0,
                extra_headers={
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "anthropic-metadata": json.dumps({
                        "task_type": "attribute_extraction",
                        "requires_consistency": True
                    })
                }
            )

            # Calculate and log costs
            cost_data = calculate_cost(response, "claude-3-5-sonnet-20240620")
            logger.info(
                f"Cost for {product_id}: "
                f"Cache {cost_data['cache_status']} | "
                f"Input: {cost_data['input_tokens']}tok (${cost_data['input_cost']:.6f}) | "
                f"Output: {cost_data['output_tokens']}tok (${cost_data['output_cost']:.6f}) | "
                f"Total: ${cost_data['total_cost']:.6f}"
            )

            # Log token usage metrics
            logger.info(
                f"Token usage for product {product_id}: "
                f"Cache creation: {response.usage.cache_creation_input_tokens if hasattr(response.usage, 'cache_creation_input_tokens') else 'N/A'}, "
                f"Cache read: {response.usage.cache_read_input_tokens if hasattr(response.usage, 'cache_read_input_tokens') else 'N/A'}, "
                f"Output: {response.usage.output_tokens}"
            )
        except Exception as e:
            logger.error(f"Failed to get response from Anthropic for product {product_id}: {str(e)}", exc_info=True)
            return product_id, None

        inference_time = time.time() - start_time
        
        # Extract JSON from response
        response_text = response.content[0].text if response.content else ""
        json_match = re.search(r'<json>(.*?)</json>', response_text, re.DOTALL)
        if not json_match:
            print(f"No JSON found in response for product {product_id}")
            return product_id, None

        try:
            result = json.loads(json_match.group(1))
            blouse_attrs = BlouseAttributes(**result)
            
            # Generate text embedding
            try:
                embedding = call_cohere_api(
                    COHERE_CLIENT,
                    'embed',
                    texts=[blouse_attrs.search_context],
                    model="embed-english-v3.0",
                    input_type="search_document",
                    embedding_types=["float"]
                ).embeddings.float[0]
            except Exception as e:
                logger.error(f"Failed to generate text embedding for product {product_id}: {str(e)}", exc_info=True)
                return product_id, None
            
            result = blouse_attrs.model_dump()
            result['text_embedding'] = embedding
            result['image_embedding'] = image_embedding
            result['image_urls'] = image_urls
            result['product_url'] = product_url if pd.notna(product_url) else ""
            result.update({
                'processed_at': pd.Timestamp.now(),
                'inference_time': inference_time,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            })
            return product_id, result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for product {product_id}: {str(e)}")
            return product_id, None

    except Exception as e:
        logger.error(
            f"Error processing product {row.get('Product ID', 'unknown')}: {str(e)}", 
            exc_info=True,
            extra={
                'product_id': row.get('Product ID', 'unknown'),
                'image_path': row.get('image_path', 'unknown'),
                'error_type': type(e).__name__
            }
        )
        return str(row.get('Product ID', 'unknown')), None

def load_and_prepare_data():
    df = pd.read_csv('/Users/aswinsreenivas/Projects/D2C Search Engine/uplyft_server/uplyft-shopify-scraper/data/02_intermediate/processed_dataframe.csv')
    columns = ['Brand Name', 'Product Name', 'Price', 'Product Description', 'Product Type', 'Size']
    
    # Create combined texts dictionary
    combined_texts = df[columns].apply(
        lambda row: {col: str(row[col]) for col in columns if pd.notna(row[col])}, 
        axis=1
    ).to_dict()
    
    return df, combined_texts

def copy_image(src_path: str, dest_folder: str) -> str:
    """Copy image to destination folder and return new path."""
    if not os.path.exists(src_path):
        print(f"Warning: Image not found at {src_path}")
        return None
        
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, filename)
    shutil.copy2(src_path, dest_path)
    return dest_path

def extract_json(text: str) -> dict:
    """Extract and parse JSON data from the response text."""
    try:
        json_str = text.split('<json>')[1].split('</json>')[0]
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError):
        raise ValueError("Failed to extract valid JSON from response")

if __name__ == "__main__":
    asyncio.run(main()) 