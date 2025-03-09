from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional, Set
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified manager for product configurations."""
    
    _instance = None
    _config_dir = Path(__file__).parent.parent.parent / 'config'
    _base_config: Dict[str, Any] = {}
    _product_configs: Dict[str, Dict[str, Any]] = {}
    _merged_configs: Dict[str, Dict[str, Any]] = {}
    _product_types: Set[str] = set()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Don't initialize immediately to avoid circular imports
            cls._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            
        # Initialize if not already done
        if not cls._initialized:
            cls._load_configs()
            cls._initialized = True
            
        return cls._instance
    
    @classmethod
    def _load_configs(cls) -> None:
        """Load all configuration files."""
        try:
            # Load base config first
            base_config_path = cls._config_dir / 'base_config.yaml'
            if base_config_path.exists():
                with open(base_config_path, 'r') as f:
                    cls._base_config = yaml.safe_load(f)
            else:
                logger.error(f"Base config file not found: {base_config_path}")
                cls._base_config = {}
            
            # Load product-specific configs
            for config_file in cls._config_dir.glob('*_config.yaml'):
                if config_file.name == 'base_config.yaml':
                    continue
                
                product_type = config_file.stem.replace('_config', '')
                cls._product_types.add(product_type)
                
                try:
                    with open(config_file, 'r') as f:
                        product_config = yaml.safe_load(f)
                        cls._product_configs[product_type] = product_config
                except (yaml.YAMLError, IOError) as e:
                    logger.error(f"Error loading config for {product_type}: {e}")
                    cls._product_configs[product_type] = {}
            
            logger.info(f"Loaded configurations for products: {cls._product_types}")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    @classmethod
    def get_base_config(cls) -> Dict[str, Any]:
        """Get the base configuration."""
        if not cls._base_config:
            # Ensure configs are loaded
            cls.get_instance()
        return cls._base_config
    
    @classmethod
    def get_product_config(cls, product_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific product type.
        
        For backward compatibility, this returns the merged configuration
        (base + product-specific) to match the behavior of the old ConfigLoader.
        """
        return cls.get_merged_config(product_type)
    
    @classmethod
    def get_raw_product_config(cls, product_type: str) -> Dict[str, Any]:
        """Get the raw (unmerged) configuration for a specific product type."""
        if product_type not in cls._product_configs:
            logger.warning(f"No configuration found for product type: {product_type}")
            return {}
        return cls._product_configs[product_type]
    
    @classmethod
    @lru_cache(maxsize=32)
    def get_merged_config(cls, product_type: str) -> Dict[str, Any]:
        """Get merged configuration (base + product-specific)."""
        # Check cache first
        if product_type in cls._merged_configs:
            return cls._merged_configs[product_type]
        
        # Get configs
        base_config = cls.get_base_config()
        product_config = cls.get_raw_product_config(product_type)
        
        # Handle 'extends' for backward compatibility
        if 'extends' in product_config:
            extends_path = cls._config_dir / product_config['extends']
            if extends_path.exists():
                try:
                    with open(extends_path, 'r') as f:
                        extends_config = yaml.safe_load(f)
                        # Merge extends config with base config
                        base_config = cls._deep_merge(base_config.copy(), extends_config)
                except Exception as e:
                    logger.error(f"Error loading extends config: {e}")
        
        # Deep merge the configurations
        merged_config = cls._deep_merge(base_config.copy(), product_config)
        
        # Cache the result
        cls._merged_configs[product_type] = merged_config
        return merged_config
    
    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            # Skip the 'extends' key for backward compatibility
            if key == 'extends':
                continue
                
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    @classmethod
    def get_available_product_types(cls) -> List[str]:
        """Get list of available product types."""
        return list(cls._product_types)
    
    @classmethod
    def get_attribute_config(cls, product_type: str, attribute_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific attribute of a product."""
        merged_config = cls.get_merged_config(product_type)
        attributes = merged_config.get('attributes', {})
        return attributes.get(attribute_name)
    
    @classmethod
    def add_product_type(cls, product_type: str, config: Dict[str, Any]) -> bool:
        """Add a new product type configuration."""
        try:
            # Create config file
            config_path = cls._config_dir / f'{product_type}_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update in-memory config
            cls._product_types.add(product_type)
            cls._product_configs[product_type] = config
            
            # Clear merged config cache for this product type
            if product_type in cls._merged_configs:
                del cls._merged_configs[product_type]
            
            logger.info(f"Added new product type: {product_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product type {product_type}: {e}")
            return False
    
    @classmethod
    def update_product_config(cls, product_type: str, config: Dict[str, Any]) -> bool:
        """Update configuration for an existing product type."""
        if product_type not in cls._product_types:
            logger.warning(f"Product type not found: {product_type}")
            return False
            
        try:
            # Update config file
            config_path = cls._config_dir / f'{product_type}_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update in-memory config
            cls._product_configs[product_type] = config
            
            # Clear merged config cache for this product type
            if product_type in cls._merged_configs:
                del cls._merged_configs[product_type]
            
            logger.info(f"Updated configuration for product type: {product_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating config for {product_type}: {e}")
            return False
    
    @classmethod
    def get_search_context_config(cls, product_type: str) -> Dict[str, Any]:
        """
        Get the search context configuration for a product type.
        
        Args:
            product_type: The product type (e.g., 'saree', 'blouse', 'kurta')
            
        Returns:
            Dict containing the search context configuration
        """
        # Get merged product config
        product_config = cls.get_merged_config(product_type)
        
        # Return the old-style config if it exists for backward compatibility
        if 'search_context' in product_config:
            return product_config['search_context']
            
        # Otherwise, build the config from the new attribute structure
        search_context_config = {
            'include_base_attributes': [],
            'include_product_attributes': []
        }
        
        # Extract attribute names for search context
        if 'attributes' in product_config:
            for attr_name, attr_config in product_config['attributes'].items():
                # Skip system fields from search context
                if attr_name in [
                    'validation_errors', 
                    'text_embedding', 
                    'search_context'
                ]:
                    continue
                    
                # Check if the attribute should be included in search context
                if (isinstance(attr_config, dict) and 
                    attr_config.get('in_search_context', False)):
                    search_context_config['include_product_attributes'].append(
                        attr_name
                    )
        
        return search_context_config
    
    @classmethod
    def get_required_attributes(cls, product_type: str) -> List[str]:
        """
        Get a list of required attributes for a product type.
        
        Args:
            product_type: The product type (e.g., 'saree', 'blouse', 'kurta')
            
        Returns:
            List of required attribute names
        """
        product_config = cls.get_merged_config(product_type)
        required_attrs = []
        
        if 'attributes' in product_config:
            for attr_name, attr_config in product_config['attributes'].items():
                if (isinstance(attr_config, dict) and 
                    attr_config.get('required', False)):
                    required_attrs.append(attr_name)
        
        return required_attrs


# For backward compatibility
class ProductConfigManager(ConfigManager):
    """Legacy class for backward compatibility."""
    pass


class ConfigLoader(ConfigManager):
    """Legacy class for backward compatibility."""
    pass 