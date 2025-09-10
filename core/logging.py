import logging
import queue
from logging.handlers import QueueHandler

def setup_logging(log_level: str = "INFO"):
    """
    Set up logging for the application.

    This function configures the root logger to output messages to both the
    terminal and a queue that can be used to display logs in the GUI.

    Args:
        log_level (str): The minimum logging level to capture.

    Returns:
        queue.Queue: A queue that will receive all log records.
    """
    log_queue = queue.Queue()
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Terminal handler
    terminal_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    terminal_handler.setFormatter(formatter)
    
    # GUI queue handler
    queue_handler = QueueHandler(log_queue)
    
    # Add handlers to the root logger
    root_logger.addHandler(terminal_handler)
    root_logger.addHandler(queue_handler)
    
    return log_queue 