from enum import Enum
from typing import Set


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

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


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

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Pattern(str, Enum):
    ABSTRACT = "Abstract"
    ANIMAL = "Animal"
    BANDHANI = "Bandhani"
    BATIK = "Batik"
    CHECKED = "Checked"
    COLOR_BLOCK = "Color Block"
    DIGITAL_PRINT = "Digital Print"
    EMBROIDERED = "Embroidered"
    FLORAL = "Floral"
    FOIL_PRINT = "Foil Print"
    GEOMETRIC = "Geometric"
    HAND_PAINTED = "Hand-painted"
    IKAT = "Ikat"
    JAMDANI = "Jamdani"
    LEHERIYA = "Leheriya"
    OMBRE = "Ombre"
    PAISLEY = "Paisley"
    POLKA_DOT = "Polka Dot"
    SCREEN_PRINT = "Screen Print"
    SEQUINED = "Sequined"
    SHIBORI = "Shibori"
    SOLID = "Solid"
    STRIPED = "Striped"
    TIE_DYE = "Tie-dye"
    TRADITIONAL_MOTIF = "Traditional Motif"
    WOVEN = "Woven"
    ZARI = "Zari"
    TEXTURED = "Textured"
    JACQUARD = "Jacquard"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


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

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class EmbellishmentLevel(str, Enum):
    NONE = "None"
    LIGHT = "Light"
    MEDIUM = "Medium"
    HEAVY = "Heavy"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Embellishment(str, Enum):
    STONE = "Stone"
    ZARDOZI = "Zardozi"
    MIRROR_WORK = "Mirror Work"
    GOTA_PATTI = "Gota Patti"
    AARI_WORK = "Aari Work"
    BEADS = "Beads"
    CHIKANKARI = "Chikankari"
    EMBROIDERY = "Embroidery"
    MUKAISH = "Mukaish"
    PHULKARI = "Phulkari"
    SCHIFFLI = "Schiffli"
    SEQUINNED = "Sequinned"
    ZARI = "Zari"
    NONE = "None"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Occasion(str, Enum):
    WEDDING_WEAR = "Wedding Wear"
    FESTIVE_WEAR = "Festive Wear"
    CASUAL_WEAR = "Casual Wear"
    PARTY_WEAR = "Party Wear"
    WORK_WEAR = "Work Wear"
    FORMAL_EVENTS = "Formal Events"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Style(str, Enum):
    CASUAL = "Casual"
    FESTIVE = "Festive"
    INDO_WESTERN = "Indo-Western"
    MODERN_TRADITIONAL = "Modern Traditional"
    ETHNIC_FUSION = "Ethnic Fusion"
    MINIMALIST = "Minimalist"
    BOHEMIAN = "Bohemian"
    VINTAGE = "Vintage"
    ELEGANT = "Elegant"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Gender(str, Enum):
    WOMEN = "Women"
    MEN = "Men"
    UNISEX = "Unisex"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class AgeGroup(str, Enum):
    ADULT = "Adult"
    TEEN = "Teen"
    KIDS = "Kids"
    SENIOR = "Senior"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class SareeType(str, Enum):
    BANARASI = "Banarasi"
    BOMKAI_SILK = "Bomkai Silk"
    MYSORE_SILK = "Mysore Silk"
    MUGA = "Muga"
    TUSSAR = "Tussar"
    KHADI = "Khadi"
    MANGALAGIRI = "Mangalagiri"
    NARAYANPETH = "Narayanpeth"
    VENKATGIRI = "Venkatgiri"
    ILKAL = "Ilkal"
    CHANDERI = "Chanderi"
    JAMDANI = "Jamdani"
    PAITHANI = "Paithani"
    PATOLA = "Patola"
    SAMBALPURI = "Sambalpuri"
    BAGRU = "Bagru"
    BALUCHARI = "Baluchari"
    BANDHANI = "Bandhani"
    BHAGALPURI = "Bhagalpuri"
    DHARAMAVARAM = "Dharamavaram"
    GADWAL = "Gadwal"
    KANJIVARAM = "Kanjivaram"
    KOTA = "Kota"
    MAHESHWARI = "Maheshwari"
    POCHAMPALLY = "Pochampally"
    BLOCK_PRINT = "Block Print"
    IKAT = "Ikat"
    LEHERIYA = "Leheriya"
    SUNGUDI = "Sungudi"
    TAANT = "Taant"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class BorderWidth(str, Enum):
    NARROW = "Narrow"
    MEDIUM = "Medium"
    WIDE = "Wide"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class BorderDesign(str, Enum):
    CONTRAST = "Contrast"
    EMBROIDERED = "Embroidered"
    PLAIN = "Plain"
    PRINTED = "Printed"
    ZARI = "Zari"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class PalluDesign(str, Enum):
    CONTRAST = "Contrast"
    ELABORATE = "Elaborate"
    MATCHING = "Matching"
    PLAIN = "Plain"
    TRADITIONAL = "Traditional"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Closure(str, Enum):
    BACK_HOOK = "Back Hook"
    FRONT_HOOK = "Front Hook"
    BACK_ZIPPER = "Back Zipper"
    SIDE_ZIPPER = "Side Zipper"
    FRONT_ZIP = "Front Zip"
    TIE_BACK = "Tie-Back"
    PULL_ON = "Pull-On"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Fit(str, Enum):
    FITTED = "Fitted"
    REGULAR = "Regular"
    RELAXED = "Relaxed"
    SLIM = "Slim"
    FORM_FITTING = "Form-Fitting"
    CROP = "Crop"
    BOXY = "Boxy"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


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

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class Hemline(str, Enum):
    STRAIGHT = "Straight"
    CURVED = "Curved"
    ASYMMETRIC = "Asymmetric"
    CROPPED = "Cropped"
    HIGH_LOW = "High-Low"
    PEPLUM = "Peplum"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class SleeveType(str, Enum):
    SLEEVELESS = "Sleeveless"
    SHORT_SLEEVES = "Short Sleeves"
    THREE_QUARTER_SLEEVES = "Three-Quarter Sleeves"
    FULL_SLEEVES = "Full Sleeves"
    ELBOW_LENGTH = "Elbow-Length"
    SPAGHETTI_STRAPS = "Spaghetti Straps"
    OFF_SHOULDER = "Off-shoulder"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls}


class KurtaSet(str, Enum):
    KURTA_ONLY = "Kurta Only"
    KURTA_WITH_DUPATTA = "Kurta with Dupatta"
    KURTA_WITH_PALAZZO = "Kurta with Palazzo"
    KURTA_WITH_PANTS = "Kurta with Pants"
    COMPLETE_SET = "Complete Set (Kurta + Bottom + Dupatta)"
    OTHERS = "Others"

    @classmethod
    def allowed_values(cls) -> Set[str]:
        return {item.value for item in cls} 