from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional, Set, Union, Tuple
import logging
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager."""
    
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
                
                # Store product type in lowercase for case-insensitive lookup
                product_type = config_file.stem.replace('_config', '').lower()
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
        
        Returns the merged configuration (base + product-specific).
        """
        return cls.get_merged_config(product_type)
    
    @classmethod
    def get_raw_product_config(cls, product_type: str) -> Dict[str, Any]:
        """Get the raw (unmerged) configuration for a specific product type."""
        # Convert to lowercase for case-insensitive lookup
        product_type = product_type.lower()
        if product_type not in cls._product_configs:
            logger.warning(f"No configuration found for product type: {product_type}")
            return {}
        return cls._product_configs[product_type]
    
    @classmethod
    @lru_cache(maxsize=32)
    def get_merged_config(cls, product_type: str) -> Dict[str, Any]:
        """Get merged configuration (base + product-specific)."""
        # Convert to lowercase for case-insensitive lookup
        product_type = product_type.lower()
        
        # Ensure configs are loaded
        cls.get_instance()
        
        # Get base config
        base_config = cls.get_base_config()
        
        # Get product-specific config
        if product_type not in cls._product_configs:
            logger.warning(f"No configuration found for product type: {product_type}")
            return base_config.copy()  # Return just base config if product config not found
        
        product_config = cls._product_configs[product_type]
        
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
        
        # Deep merge base config with product config
        merged_config = cls._deep_merge(base_config.copy(), product_config)
        return merged_config
    
    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()  # Create a new copy to avoid modifying the original
        
        for key, value in override.items():
            # Skip the 'extends' key for backward compatibility
            if key == 'extends':
                continue
                
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = cls._deep_merge(result[key], value)
            else:
                # For non-dict values or new keys, just override/add
                result[key] = value
                
        return result
    
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
    def get_attribute_data_type(cls, product_type: str, attribute_name: str) -> Optional[str]:
        """Get the data_type for a specific attribute."""
        attr_config = cls.get_attribute_config(product_type, attribute_name)
        if not attr_config:
            return None
        
        return attr_config.get('data_type')
    
    @classmethod
    def get_attribute_item_type(cls, product_type: str, attribute_name: str) -> Optional[str]:
        """Get the item_type for a specific attribute."""
        attr_config = cls.get_attribute_config(product_type, attribute_name)
        if not attr_config:
            return None
        
        return attr_config.get('item_type')
    
    @classmethod
    def is_enum_attribute(cls, product_type: str, attribute_name: str) -> bool:
        """Check if an attribute is an enum type."""
        attr_config = cls.get_attribute_config(product_type, attribute_name)
        if not attr_config:
            return False
        
        # Check if it's directly an enum
        if attr_config.get('data_type') == 'string' and attr_config.get('item_type') == 'enum':
            return True
        
        # Check if it's a list of enums
        if attr_config.get('data_type') == 'list' and attr_config.get('item_type') == 'enum':
            return True
        
        return False
    
    @classmethod
    def get_enum_values(cls, product_type: str, attribute_name: str) -> Optional[str]:
        """Get the enum class name for an enum attribute."""
        attr_config = cls.get_attribute_config(product_type, attribute_name)
        if not attr_config:
            return None
        
        return attr_config.get('allowed_values')
    
    @classmethod
    def get_attribute_threshold(cls, product_type: str, attribute_name: str) -> float:
        """Get the threshold for an enum attribute."""
        attr_config = cls.get_attribute_config(product_type, attribute_name)
        if not attr_config:
            return 0.8  # Default threshold
        
        return attr_config.get('threshold', 0.8)
    
    @classmethod
    def validate_attribute_config(cls, attr_config: Dict[str, Any]) -> List[str]:
        """
        Validate an attribute configuration against the standardized structure.
        
        Args:
            attr_config: The attribute configuration to validate
            
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Check for required fields
        if 'required' not in attr_config:
            errors.append("Missing 'required' field")
        
        if 'data_type' not in attr_config:
            errors.append("Missing 'data_type' field")
        else:
            data_type = attr_config['data_type']
            
            # All attributes should have item_type
            if 'item_type' not in attr_config:
                errors.append("Missing 'item_type' field")
            else:
                item_type = attr_config['item_type']
                
                # Validate based on data_type and item_type
                if data_type == 'list':
                    # For enum lists, check for allowed_values
                    if item_type == 'enum' and 'allowed_values' not in attr_config:
                        errors.append("Enum list missing 'allowed_values'")
                    
                    # For enum lists, check for threshold
                    if item_type == 'enum' and 'threshold' not in attr_config:
                        errors.append("Enum list missing 'threshold'")
                
                elif data_type == 'string' and item_type == 'enum':
                    # Check for allowed_values
                    if 'allowed_values' not in attr_config:
                        errors.append("Enum attribute missing 'allowed_values'")
                    
                    # Check for threshold
                    if 'threshold' not in attr_config:
                        errors.append("Enum attribute missing 'threshold'")
                
                elif data_type == 'object':
                    # Check for properties
                    if 'properties' not in attr_config:
                        errors.append("Object attribute missing 'properties'")
        
        # Check for in_search_context
        if 'in_search_context' not in attr_config:
            errors.append("Missing 'in_search_context' field")
        
        return errors
    
    @classmethod
    def validate_product_config(cls, product_type: str) -> Dict[str, List[str]]:
        """
        Validate a product configuration against the standardized structure.
        
        Args:
            product_type: The product type to validate
            
        Returns:
            Dictionary mapping attribute names to lists of validation errors
        """
        merged_config = cls.get_merged_config(product_type)
        validation_results = {}
        
        if 'attributes' not in merged_config:
            return {'_global': ['Missing attributes section']}
        
        for attr_name, attr_config in merged_config['attributes'].items():
            errors = cls.validate_attribute_config(attr_config)
            if errors:
                validation_results[attr_name] = errors
        
        return validation_results
    
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
    
    def __init_subclass__(cls) -> None:
        warnings.warn(
            "Subclassing ConfigManager is deprecated. Use ConfigManager directly.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init_subclass__() 