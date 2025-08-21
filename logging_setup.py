# logging_setup.py
import logging, os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file="logs/app.log", level="INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers on reload
    for h in list(root.handlers):
        root.removeHandler(h)

    file_h = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=7, encoding="utf-8")
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # show logs in console only outside production
    if os.getenv("FLASK_ENV") != "production":
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)