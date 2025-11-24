"""
Utility helper functions for logging, file operations, etc.
"""

import logging
from pathlib import Path


def setup_logger(log_file: str = "logs/app.log") -> logging.Logger:
    """Configures and returns a logger instance."""
        Path("logs").mkdir(exist_ok=True)

            logging.basicConfig(
                    filename=log_file,
                            level=logging.INFO,
                                    format="%(asctime)s — %(levelname)s — %(message)s",
                                        )
                                            return logging.getLogger("customer_churn")