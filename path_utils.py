# path_utils.py
import os
import sys


def get_project_root():
    """Get the project root"""
    # Method 1: read from an environment variable
    project_root = os.environ.get("MEDICAL_PIPELINE_ROOT")
    if project_root and os.path.exists(project_root):
        return project_root

    # Method 2: infer from the current file location
    current_file = os.path.abspath(__file__)
    # Assume this file is in the project root
    return os.path.dirname(current_file)


def get_relative_path(*path_parts):
    """Get a path relative to the project root"""
    root = get_project_root()
    return os.path.join(root, *path_parts)


def ensure_dir_exists(dir_path):
    """Ensure the directory exists"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
