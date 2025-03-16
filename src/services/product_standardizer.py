from typing import Dict, Any
import logging
from src.base.clothing_item import ClothingItem
import pandas as pd

logger = logging.getLogger(__name__)

class ProductStandardizer:
    @staticmethod
    def standardize_product(product: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize product data with category mapping"""
        try:
            original_category = product.get('product_type', '')
            # Handle missing product_type
            if pd.isna(original_category):
                original_category = ''
            # Convert to title case before standardization
            original_category = original_category.title()
            standardized = ClothingItem.standardize_category(original_category)
            
            return {
                **product,
                'original_category': original_category,
                'standardized_category': standardized,
                'search_category': standardized.lower().replace(' ', '_')
            }
            
        except Exception as e:
            logger.error(
                f"Error standardizing product {product.get('id')}: {e}"
            )
            return {
                **product,
                'original_category': original_category,
                'standardized_category': 'Uncategorized',
                'search_category': 'uncategorized'
            }

    @staticmethod
    def batch_standardize(products: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Process multiple products with error handling"""
        return [ProductStandardizer.standardize_product(p) for p in products] 