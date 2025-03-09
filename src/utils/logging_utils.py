import logging
import os
import sys
import platform
import socket
import time
import json
import psutil
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union


def get_system_info() -> Dict[str, Any]:
    """Get detailed system information for logging context."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "username": os.getlogin(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_percent": disk.percent
        }
    except Exception as e:
        # Fallback to basic system info if psutil fails
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "username": os.getlogin(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "error_getting_detailed_info": str(e)
        }


class JsonFormatter(logging.Formatter):
    """Format log records as JSON strings."""
    
    def __init__(self, include_timestamp: bool = True):
        self.include_timestamp = include_timestamp
        super().__init__()
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)
            
        return json.dumps(log_data)


class PerformanceMetrics:
    """Track performance metrics during execution."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.durations = {}
        
    def checkpoint(self, name: str):
        """Record a checkpoint with the current time."""
        self.checkpoints[name] = time.time()
        
    def measure(self, name: str, start_point: str = None, end_point: str = None):
        """Measure duration between checkpoints or from start."""
        if start_point and end_point:
            if start_point in self.checkpoints and end_point in self.checkpoints:
                duration = self.checkpoints[end_point] - self.checkpoints[start_point]
                self.durations[f"{start_point}_to_{end_point}"] = duration
                return duration
        elif start_point:
            if start_point in self.checkpoints:
                duration = time.time() - self.checkpoints[start_point]
                self.durations[f"{start_point}_to_now"] = duration
                return duration
        else:
            # Measure from start
            duration = time.time() - self.start_time
            self.durations["total"] = duration
            return duration
        
        return None
    
    def get_report(self) -> Dict[str, float]:
        """Get a report of all durations."""
        # Update total duration
        self.durations["total"] = time.time() - self.start_time
        return {k: round(v, 3) for k, v in self.durations.items()}


def setup_logging(
    app_name: str,
    log_level: Union[int, str] = logging.INFO,
    log_dir: Optional[Path] = None,
    console: bool = True,
    include_system_info: bool = True,
    max_log_files: int = 30,
    max_file_size_mb: int = 10,
    json_format: bool = False,
    log_modules: Optional[List[str]] = None
) -> Path:
    """
    Set up enhanced logging configuration for the application.
    
    Args:
        app_name: Name of the application (used in log filename)
        log_level: Logging level (default: INFO)
        log_dir: Directory to store logs (default: logs/ in project root)
        console: Whether to log to console as well (default: True)
        include_system_info: Whether to log system information (default: True)
        max_log_files: Maximum number of log files to keep (default: 30)
        max_file_size_mb: Maximum size of each log file in MB (default: 10)
        json_format: Whether to use JSON format for logs (default: False)
        log_modules: List of module names to set specific log levels for
        
    Returns:
        Path to the log file
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    
    # Configure timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{app_name}_{timestamp}.log"
    
    # Set up handlers
    handlers = []
    
    # File handler with rotation
    if max_file_size_mb > 0:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(log_level)
    handlers.append(file_handler)
    
    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    # Configure logging format
    if json_format:
        formatter = JsonFormatter()
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
    
    # Apply formatter to handlers
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set specific log levels for modules if provided
    if log_modules:
        for module_config in log_modules:
            if ":" in module_config:
                module_name, module_level = module_config.split(":", 1)
                module_level = getattr(logging, module_level.upper(), logging.INFO)
                logging.getLogger(module_name).setLevel(module_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {app_name} - Log file: {log_file}")
    
    # Log system information
    if include_system_info:
        sys_info = get_system_info()
        logger.info("System information:")
        for key, value in sys_info.items():
            logger.info(f"  {key}: {value}")
        
        # Log command line arguments
        logger.info(f"Command line: {' '.join(sys.argv)}")
        
        # Log environment variables (excluding sensitive ones)
        env_vars = {
            k: v for k, v in os.environ.items() 
            if not any(
                sensitive in k.lower() 
                for sensitive in ['key', 'secret', 'password', 'token', 'credential']
            )
        }
        logger.info(f"Environment variables: {len(env_vars)} variables set")
        
        # Log Python packages
        try:
            import pkg_resources
            installed_packages = sorted([
                f"{pkg.key}=={pkg.version}" 
                for pkg in pkg_resources.working_set
            ])
            logger.info(f"Installed packages: {len(installed_packages)} packages")
            logger.debug(f"Packages: {', '.join(installed_packages)}")
        except Exception as e:
            logger.warning(f"Could not retrieve installed packages: {e}")
    
    return log_file


def log_execution_time(logger, start_time: float, operation: str):
    """Log the execution time of an operation."""
    elapsed = time.time() - start_time
    logger.info(f"{operation} completed in {elapsed:.2f} seconds")


class LoggingContext:
    """Context manager for logging execution time and status."""
    
    def __init__(self, logger, operation: str, log_level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.start_time = 0.0
        self.log_level = log_level
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(
                self.log_level, 
                f"{self.operation} completed successfully in {elapsed:.2f} seconds"
            )
        else:
            self.logger.error(
                f"{self.operation} failed after {elapsed:.2f} seconds: {exc_val}"
            )
        return False  # Don't suppress exceptions


class StructuredLogger:
    """Helper class for structured logging with consistent fields."""
    
    def __init__(self, logger_name: str, extra_fields: Dict[str, Any] = None):
        self.logger = logging.getLogger(logger_name)
        self.extra_fields = extra_fields or {}
        
    def _log(self, level: int, msg: str, extra: Dict[str, Any] = None, **kwargs):
        """Log with extra fields merged."""
        all_extras = self.extra_fields.copy()
        if extra:
            all_extras.update(extra)
            
        # Create a record manually to add extra fields
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=kwargs.get('exc_info'),
        )
        
        # Add extra fields to the record
        record.extra = all_extras
        
        # Pass to handlers directly
        for handler in self.logger.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)
    
    def log(self, level: int, msg: str, **kwargs):
        """Log at specified level - added for compatibility with LoggingContext."""
        extra = kwargs.pop('extra', None)
        self._log(level, msg, extra, **kwargs)
    
    def debug(self, msg: str, extra: Dict[str, Any] = None, **kwargs):
        self._log(logging.DEBUG, msg, extra, **kwargs)
    
    def info(self, msg: str, extra: Dict[str, Any] = None, **kwargs):
        self._log(logging.INFO, msg, extra, **kwargs)
    
    def warning(self, msg: str, extra: Dict[str, Any] = None, **kwargs):
        self._log(logging.WARNING, msg, extra, **kwargs)
    
    def error(self, msg: str, extra: Dict[str, Any] = None, **kwargs):
        self._log(logging.ERROR, msg, extra, **kwargs)
    
    def critical(self, msg: str, extra: Dict[str, Any] = None, **kwargs):
        self._log(logging.CRITICAL, msg, extra, **kwargs) 