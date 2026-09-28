import json
import logging
import time
from contextlib import contextmanager
from typing import Generator


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        })


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def latency_tracker(operation: str, logger: logging.Logger | None = None) -> Generator[dict, None, None]:
    """Context manager that tracks execution time in milliseconds."""
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = elapsed_ms
        log = logger or get_logger("latency")
        log.info(f"{operation} completed in {elapsed_ms:.2f}ms")
