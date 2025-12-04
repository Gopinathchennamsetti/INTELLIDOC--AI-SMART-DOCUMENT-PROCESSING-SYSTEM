"""
config.py

Configuration constants and directory setup for the Flask application.
"""

import os
import logging

# Folder paths
UPLOAD_FOLDER = 'uploads'
CLUSTERED_FOLDER = 'clustered_output'
CONSOLIDATED_FOLDER = 'consolidated_output'
DUPLICATE_FOLDER = 'duplicate_output'
SUMMARY_FOLDER = 'summary_output'

# Secret key for Flask session management
SECRET_KEY = 'secret1237'


def create_folder_if_not_exists(folder_path: str):
    """Create a folder if it does not exist and log the operation."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f'Created folder: {folder_path}')
        else:
            logging.info(f'Folder already exists: {folder_path}')
    except Exception as e:
        logging.error(f'Error creating folder {folder_path}: {e}')


# Ensure required folders exist
create_folder_if_not_exists(UPLOAD_FOLDER)
create_folder_if_not_exists(CLUSTERED_FOLDER)
create_folder_if_not_exists(CONSOLIDATED_FOLDER)
create_folder_if_not_exists(DUPLICATE_FOLDER)
create_folder_if_not_exists(SUMMARY_FOLDER)