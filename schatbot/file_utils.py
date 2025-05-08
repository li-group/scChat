import os
import shutil
import json
import regex as re
import logging

logger = logging.getLogger(__name__)

def clear_directory(directory_path: str) -> None:
    """
    Clears all files and subdirectories within the given directory.
    """
    if not os.path.exists(directory_path):
        logger.info(f"Directory {directory_path} does not exist.")
        return
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")

def find_file_with_extension(directory_path: str, extension: str) -> str:
    """
    Recursively searches for a file with the given extension.
    Returns the full path if found; otherwise, returns an empty string.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return ""

def find_and_load_sample_mapping(directory: str):
    """
    Searches for 'sample_mapping.json' in the given directory tree and loads it.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    sample_mapping = json.load(f)
                logger.info(f"'sample_mapping.json' found and loaded from {file_path}")
                return sample_mapping
    return None

def safe_filename(term: str) -> str:
    """
    Replaces non-alphanumeric characters with underscores and truncates the result to 50 characters.
    """
    term_safe = re.sub(r'[^a-zA-Z0-9_]', '_', term)
    return term_safe[:50]