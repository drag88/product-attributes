import asyncio
import logging
import os
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
import argparse
from datetime import datetime

from src.factories.clothing_factory import ClothingFactory
from src.services.api_service import APIService
from src.services.attribute_generator import AttributeGenerator
from src.services.batch_processor import BatchProcessor
from src.utils.logging_utils import (
    setup_logging, 
    LoggingContext, 
    PerformanceMetrics,
    StructuredLogger
)
from src.services.product_standardizer import ProductStandardizer
from src.base.clothing_item import ClothingItem

# Load environment variables
load_dotenv()

# Set up logging with enhanced features
log_file = setup_logging(
    app_name="product_attributes",
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    max_file_size_mb=20,
    max_log_files=50,
    json_format=os.environ.get("LOG_JSON", "").lower() == "true",
    log_modules=[
        "src.services.api_service:DEBUG",
        "src.services.attribute_generator:DEBUG"
    ]
)

# Initialize performance metrics
perf_metrics = PerformanceMetrics()
perf_metrics.checkpoint("program_start")

# Initialize structured logger
logger = StructuredLogger(
    __name__,
    extra_fields={"component": "main", "version": "1.0.0"}
)
logger.info("Starting product attribute generation")


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
        "Brand Name": "brand_name",
        "Product ID": "product_id", 
        "Product Type": "product_type",
        "Product Name": "title",
        "Product Description": "description",
        "Product Image Link": "product_image_link",
        "Price": "price",
        "Size": "size",
        "Tags": "tags",
        "Product URL": "product_url"
    }
    # Rename columns
    df = df.rename(columns=column_mappings)
    
    # Get image paths
    image_paths = {}
    images_dir = Path(config["paths"]["input"].get("images_dir")) / config['attributes']['brand']['default']
    
    if images_dir:
        images_dir = Path(images_dir)
        if images_dir.exists():
            for product_id in df["product_id"]:
                product_id_str = str(product_id)
                potential_paths = [
                    images_dir / f"{product_id_str}.jpg",
                    images_dir / f"{product_id_str}.png",
                    images_dir / f"{product_id_str}.webp",
                    images_dir / product_id_str / "main.jpg",
                    images_dir / product_id_str / "main.png"
                ]
                
                for path in potential_paths:
                    if path.exists():
                        image_paths[product_id_str] = str(path)
                        break
    
    return df, image_paths


def filter_products(
    df: pd.DataFrame,
    available_types: List[str],
    target_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter products based on standardized product categories.
    
    Args:
        df: DataFrame containing products
        available_types: List of available standardized product types
        target_types: Optional list of target product types to filter for
        
    Returns:
        Filtered DataFrame with standardized categories
    """
    # Standardize all product types
    df['standardized_category'] = df['product_type'].apply(
        lambda x: ProductStandardizer.standardize_product({'product_type': x})['standardized_category']
    )
    
    # Standardize target types if provided
    if target_types:
        target_standardized = [
            ProductStandardizer.standardize_product({'product_type': t})['standardized_category']
            for t in target_types
        ]
        target_standardized = list(set(target_standardized))  # Remove duplicates
    else:
        target_standardized = None

    # First filter by target types if specified
    if target_standardized:
        filtered_df = df[df['standardized_category'].isin(target_standardized)]
    else:
        filtered_df = df.copy()

    # Then filter for available types
    filtered_df = filtered_df[filtered_df['standardized_category'].isin(available_types)]

    # Logging and reporting
    if not filtered_df.empty:
        # Get standardized type distribution
        type_counts = filtered_df['standardized_category'].value_counts().to_dict()
        print(f"Standardized category distribution: {type_counts}")
        logger.info(
            "Standardized category distribution",
            extra={"type_counts": type_counts}
        )
    
    return filtered_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate product attributes using AI"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of products to process in each batch"
    )
    
    parser.add_argument(
        "--product-types",
        type=str,
        nargs="+",
        help="Product types to process (default: all available types)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


async def main():
    try:
        print("Starting main function")
        perf_metrics.checkpoint("args_parsing_start")
        args = parse_args()
        perf_metrics.checkpoint("args_parsing_end")
        
        # Update log level if specified in args
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
            for handler in logging.getLogger().handlers:
                handler.setLevel(getattr(logging, args.log_level))
            logger.info(f"Set log level to {args.log_level}")
            print(f"Set log level to {args.log_level}")
        
        print("Before configuration loading")
        try:
            with LoggingContext(logger, "configuration loading"):
                print("Inside configuration loading")
                perf_metrics.checkpoint("config_loading_start")
                # Load configuration
                config = load_config(args.config)
                print(f"Loaded configuration from {args.config}")
                logger.info(
                    f"Loaded configuration from {args.config}",
                    extra={"config_path": args.config}
                )
                
                # Initialize factory
                ClothingFactory.initialize()
                raw_available_types = ClothingFactory.get_available_types()
                # Standardize the available types names to match our mapping
                available_types = [
                    ClothingItem.standardize_category(t) 
                    for t in raw_available_types.keys()
                ]
                print(f"Available product types: {', '.join(available_types)}")
                logger.info(
                    f"Available product types: {', '.join(available_types)}",
                    extra={"available_types": available_types}
                )
                perf_metrics.checkpoint("config_loading_end")
        except Exception as e:
            print(f"Error during configuration loading: {str(e)}")
            logger.error(f"Error during configuration loading: {str(e)}", exc_info=True)
            raise
        
        print("Before data loading")
        try:
            with LoggingContext(logger, "data loading"):
                print("Inside data loading")
                perf_metrics.checkpoint("data_loading_start")
                # Load data
                print(f"Loading data from {config['paths']['input']['product_data']}")
                logger.info(f"Loading data from {config['paths']['input']['product_data']}")
                df, image_paths = load_and_prepare_data(config)
                print(f"Loaded {len(df)} products")
                logger.info(
                    f"Loaded {len(df)} products",
                    extra={"product_count": len(df)}
                )
                
                # Filter for target product types
                print(f"Filtering products for types: {args.product_types}")
                logger.info(f"Filtering products for types: {args.product_types}")
                df = filter_products(df, available_types, args.product_types)
                
                if df.empty:
                    print("No products found to process")
                    logger.warning("No products found to process")
                    return
                
                print(f"Processing {len(df)} products")
                logger.info(
                    f"Processing {len(df)} products",
                    extra={"filtered_product_count": len(df)}
                )
                perf_metrics.checkpoint("data_loading_end")
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            logger.error(f"Error during data loading: {str(e)}", exc_info=True)
            raise
        
        try:
            with LoggingContext(logger, "API service initialization"):
                perf_metrics.checkpoint("service_init_start")
                # Initialize API service
                print("Initializing API service")
                api_service = APIService(config.get('api', {}))
                
                # Initialize attribute generator
                attribute_generator = AttributeGenerator(
                    api_service=api_service,
                    config=config
                )
                
                # Initialize batch processor
                batch_processor = BatchProcessor(
                    api_service=api_service,
                    attribute_generator=attribute_generator,
                    output_dir=Path(args.output_dir),
                    batch_size=args.batch_size
                )
                print(f"Batch processor initialized with batch size: {args.batch_size}")
                perf_metrics.checkpoint("service_init_end")
        except Exception as e:
            logger.error(f"Error during API service initialization: {str(e)}", exc_info=True)
            raise
        
        try:
            with LoggingContext(logger, "product processing"):
                perf_metrics.checkpoint("processing_start")
                print("Starting product processing")
                # Process each product type
                unique_categories = df['standardized_category'].unique()
                logger.info(f"Found product types in data: {unique_categories}")
                
                for category in unique_categories:
                    if args.product_types and category not in args.product_types:
                        logger.info(f"Skipping {category} as it's not in requested types: {args.product_types}")
                        continue
                    
                    logger.info(
                        f"Processing {category} products",
                        extra={"current_product_type": category}
                    )
                    
                    # Filter dataframe for current product type
                    type_df = df[df['standardized_category'] == category].copy()
                    logger.info(f"Found {len(type_df)} products of type {category}")
                    
                    # Add image paths if available
                    if image_paths:
                        type_df['image_path'] = type_df['product_id'].astype(str).map(
                            lambda x: image_paths.get(x)
                        )
                        
                        # Filter out products without images if needed
                        if config.get('require_images', False):
                            has_images = type_df['image_path'].notna().sum()
                            logger.info(f"{has_images} out of {len(type_df)} products have images")
                            type_df = type_df[type_df['image_path'].notna()]
                        
                        if type_df.empty:
                            logger.warning(
                                f"No {category} products found with valid images",
                                extra={"product_type": category}
                            )
                            continue
                            
                        logger.info(
                            f"Processing {len(type_df)} products with valid images",
                            extra={
                                "product_type": category,
                                "product_count": len(type_df)
                            }
                        )
                    
                    # Process products in batches using the batch processor
                    logger.info(f"Starting batch processing for {category}")
                    results = await batch_processor.batch_process(
                        type_df,
                        category,
                        image_path_column='image_path' if 'image_path' in type_df.columns else None
                    )
                    
                    if not results:
                        logger.warning(
                            f"No results generated for {category}",
                            extra={"product_type": category}
                        )
                    else:
                        logger.info(f"Generated results for {len(results)} products of type {category}")
                perf_metrics.checkpoint("processing_end")
        except Exception as e:
            logger.error(f"Error during product processing: {str(e)}", exc_info=True)
            raise
        
        # Export cost report
        with LoggingContext(logger, "cost reporting"):
            perf_metrics.checkpoint("cost_report_start")
            output_dir = Path(args.output_dir)
            api_service.export_cost_report(str(output_dir))
            perf_metrics.checkpoint("cost_report_end")
        
        # Log performance metrics
        perf_metrics.checkpoint("program_end")
        performance_report = perf_metrics.get_report()
        logger.info(
            "Performance metrics",
            extra={"performance": performance_report}
        )
        
        logger.info("Product attribute generation completed successfully")
        
    except Exception as e:
        logger.error(
            f"Error in main execution: {str(e)}",
            extra={"error_type": type(e).__name__},
            exc_info=True
        )
        raise


if __name__ == "__main__":
    asyncio.run(main()) 