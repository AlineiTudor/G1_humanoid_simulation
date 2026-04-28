def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create configured logger."""
    ...

def log_sensor_stats(
    logger: logging.Logger,
    sensor_data: Dict[str, Any],
    step: int,
):
    """Log statistics about sensor data."""
    ...