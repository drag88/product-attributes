#!/usr/bin/env python3
"""
CLI script to add a new product type to the system.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import yaml
from datetime import datetime
from src.utils.product_manager import ProductManager
from src.utils.logging_utils import setup_logging, LoggingContext

# Set up logging
log_file = setup_logging("add_product_type")

logger = logging.getLogger(__name__)
logger.info("Starting add_product_type")

def main():
    parser = argparse.ArgumentParser(description='Add a new product type to the system')
    parser.add_argument('product_type', help='Name of the product type to add (e.g., dress)')
    parser.add_argument('--config', help='Path to the config file', default=None)
    parser.add_argument('--template', help='Path to the template file', default=None)
    args = parser.parse_args()
    
    try:
        product_type = args.product_type.lower()
        logger.info(f"Adding new product type: {product_type}")
        
        with LoggingContext(logger, "template generation"):
            # Generate template config if not provided
            if args.config is None:
                config = ProductManager.create_product_template(product_type)
                logger.info(f"Generated template config for {product_type}")
            else:
                # Load config from file
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded config from {args.config}")
            
            # Generate template class code if not provided
            if args.template is None:
                class_code = ProductManager.generate_product_class_template(product_type)
                logger.info(f"Generated template class code for {product_type}")
            else:
                # Load template from file
                with open(args.template, 'r') as f:
                    class_code = f.read()
                logger.info(f"Loaded template from {args.template}")
        
        with LoggingContext(logger, "product type registration"):
            # Add the new product type
            success = ProductManager.add_new_product_type(
                product_type=product_type,
                config=config,
                product_class_code=class_code
            )
            
            if success:
                logger.info(f"Successfully added new product type: {product_type}")
                return 0
            else:
                logger.error(f"Failed to add product type: {product_type}")
                return 1
    
    except Exception as e:
        logger.error(f"Error adding product type: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.2f} seconds")
    sys.exit(exit_code) 