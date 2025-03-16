from .validation import find_best_enum_match
from .config_loader import ConfigManager
import warnings

__all__ = ['find_best_enum_match', 'ConfigManager']

def __getattr__(name):
    """Handle legacy imports gracefully."""
    if name in ['ProductConfigManager', 'ConfigLoader']:
        warnings.warn(
            f"{name} has been removed. Use ConfigManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return ConfigManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
