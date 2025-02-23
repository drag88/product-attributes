import asyncio
import logging
import os
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
import argparse

from src.factories.clothing_factory import ClothingFactory
from src.services.api_service import APIService
from src.services.attribute_generator import AttributeGenerator

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        
    def load(self) -> Dict[str, Any]:
        """Load base configuration from YAML file"""
        config_path = self.config_dir / "base_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_prepare_data(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    input_path = config["paths"]["input"]["product_data"]
    df = pd.read_csv(input_path)
    
    # Define column mappings
    column_mappings = {
        'Brand Name': 'brand_title',  # Maps to both brand and title
        'Product Name': 'title',
        'Product Description': 'description',
        'Price': 'price',
        'Size': 'size',
        'Care Instructions': 'care_instructions'
    }
    
    # Create combined texts dictionary with mapped field names
    combined_texts = df[column_mappings.keys()].apply(
        lambda row: {
            # Special handling for brand field
            **{'brand': str(row['Brand Name'])},
            **{
                column_mappings[col]: str(row[col]) 
                for col in column_mappings 
                if pd.notna(row[col])
            }
        }, 
        axis=1
    ).to_dict()
    
    return df, combined_texts


def filter_products(
    df: pd.DataFrame,
    available_types: List[str],
    target_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter DataFrame for specific product types."""
    # Fill NaN values with empty string to avoid float errors
    df['Product Type'] = df['Product Type'].fillna('')
    
    if target_types:
        # Validate target types
        invalid_types = set(target_types) - set(available_types)
        if invalid_types:
            raise ValueError(
                f"Invalid product types: {invalid_types}. "
                f"Available types: {available_types}"
            )
        # Filter for specified types
        product_type_mask = df['Product Type'].str.lower().apply(
            lambda x: any(t in x for t in target_types)
        )
    else:
        # Filter for all available types
        product_type_mask = df['Product Type'].str.lower().apply(
            lambda x: any(t in x for t in available_types)
        )
    
    filtered_df = df[product_type_mask]
    logger.info(
        f"Filtered {len(filtered_df)} products for types: "
        f"{target_types or 'all'}"
    )
    return filtered_df


async def process_product(
    product_data: Dict[str, Any],
    product_type: str,
    attribute_generator: AttributeGenerator
) -> Optional[Dict[str, Any]]:
    try:
        product_id = product_data.get('product_id', 'unknown')
        logger.info(
            "Starting processing for product "
            f"{product_id}"
        )

        # Convert size string to list if needed
        if 'size' in product_data and isinstance(product_data['size'], str):
            product_data['size'] = [product_data['size']]

        # Generate attributes using the API
        try:
            generated_attributes = await attribute_generator.generate_attributes(
                product_data=product_data,
                product_type=product_type
            )
        except Exception as e:
            logger.error(
                f"Failed to generate attributes for product {product_id}: "
                f"{str(e)}", 
                exc_info=True
            )
            return None

        # Merge generated attributes with input data
        result = {**product_data, **generated_attributes}

        # Validate the final data by creating a product instance
        try:
            ClothingFactory.create(product_type, **result)
            return result
        except Exception as e:
            logger.error(
                f"Failed to validate product {product_id}: {str(e)}", 
                exc_info=True
            )
            return None

    except Exception as e:
        logger.error(
            f"Error processing product {product_id}: {str(e)}", 
            exc_info=True,
            extra={
                'product_id': product_id,
                'error_type': type(e).__name__
            }
        )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate product attributes for clothing items"
    )
    parser.add_argument(
        "--product-types",
        nargs="*",
        help=(
            "Space-separated list of product types to process. "
            "If not provided, processes all available types."
        )
    )
    return parser.parse_args()


async def main():
    try:
        # Load base configuration
        config_loader = ConfigLoader(Path("config"))
        base_config = config_loader.load()
        
        # Initialize services with the config dict
        api_service = APIService(base_config)
        attribute_generator = AttributeGenerator(api_service, base_config)
        
        # Rest of processing logic
        args = parse_args()
        target_types = (
            [t.lower() for t in args.product_types] 
            if args.product_types 
            else None
        )
        
        # Load and prepare data using the same logic as original file
        df, combined_texts = load_and_prepare_data(base_config)
        
        # Get available product types
        available_types = ClothingFactory.get_available_types()
        logger.info(f"Available product types: {available_types}")
        
        # Filter for target product types
        df = filter_products(df, available_types, target_types)
        
        if df.empty:
            logger.warning("No products found to process")
            return
        
        results = []
        for _, row in df.iterrows():
            # Determine product type
            product_type = next(
                t for t in available_types 
                if t in row['Product Type'].lower()
            )
            
            # Register enum mappings for this product type
            enum_mappings = ClothingFactory.get_enum_mappings(product_type)
            for field, enum_class in enum_mappings.items():
                attribute_generator.register_enum_mapping(field, enum_class)
            
            product_data = {
                **combined_texts[row.name],
                'product_id': str(row['Product ID']),
                'image_path': row.get('image_path')
            }
            result = await process_product(
                product_data, 
                product_type,
                attribute_generator
            )
            results.append(result)

        # Save results
        output_base = Path(base_config["paths"]["output"]["base_dir"])
        attributes_path = base_config["paths"]["output"]["structure"]["attributes"]
        
        # Use appropriate output path based on target types
        product_type_str = (
            target_types[0] if target_types and len(target_types) == 1 
            else "all"
        )
        output_path = (
            output_base / attributes_path.format(product_type=product_type_str)
        )
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "attributes.json", "w") as f:
            yaml.dump(results, f)
            
        logger.info(
            f"Successfully processed {len(results)} products. "
            f"Results saved to {output_path}"
        )

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 