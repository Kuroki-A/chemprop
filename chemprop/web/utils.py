"""Contains utility functions for the Flask web app."""

import os
import shutil
from typing import Any

from flask import Flask


MIN_REMOTE_PASSWORD_LENGTH = 16
MIN_REMOTE_SECRET_KEY_BYTES = 32


def _secret_byte_length(secret: Any) -> int:
    """Returns the byte length of a Flask secret key, or zero if invalid."""
    if isinstance(secret, bytes):
        return len(secret)
    if isinstance(secret, str):
        return len(secret.encode('utf-8'))
    return 0


def validate_web_security_config(app: Flask, allow_remote: bool, debug: bool = False) -> None:
    """Validates settings which would otherwise expose the legacy web UI."""
    if not allow_remote:
        return

    if debug or app.debug:
        raise ValueError('Debug mode cannot be combined with remote access.')
    if app.config.get('ALLOW_CHECKPOINT_UPLOADS', False):
        raise ValueError('Checkpoint uploads are allowed only in loopback-only mode with trusted files.')

    password = app.config.get('WEB_PASSWORD')
    if not isinstance(password, str) or len(password) < MIN_REMOTE_PASSWORD_LENGTH:
        raise ValueError(
            'Remote access requires CHEMPROP_WEB_PASSWORD to contain at least '
            f'{MIN_REMOTE_PASSWORD_LENGTH} characters.'
        )

    if not app.config.get('SECRET_KEY_CONFIGURED', False):
        raise ValueError('Remote access requires the CHEMPROP_WEB_SECRET_KEY environment variable.')
    if _secret_byte_length(app.config.get('SECRET_KEY')) < MIN_REMOTE_SECRET_KEY_BYTES:
        raise ValueError(
            'Remote access requires CHEMPROP_WEB_SECRET_KEY to contain at least '
            f'{MIN_REMOTE_SECRET_KEY_BYTES} bytes.'
        )


def set_root_folder(app: Flask, root_folder: str = None, create_folders: bool = True) -> None:
    """
    Sets the root folder for the config along with subfolders like the data and checkpoint folders.

    :param app: Flask app.
    :param root_folder: Path to the root folder. If None, the current root folders is unchanged.
    :param create_folders: Whether to create the root folder and subfolders.
    """
    # Set root folder and subfolders
    if root_folder is not None:
        app.config['ROOT_FOLDER'] = root_folder
        app.config['DATA_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/web_data')
        app.config['CHECKPOINT_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/web_checkpoints')
        app.config['TEMP_FOLDER'] = os.path.join(app.config['ROOT_FOLDER'], 'app/temp')
        app.config['DB_PATH'] = os.path.join(app.config['ROOT_FOLDER'], app.config['DB_FILENAME'])

    # Create folders
    if create_folders:
        if not os.access(os.path.dirname(app.config['ROOT_FOLDER']), os.W_OK):
            raise ValueError(f'You do not have write permissions on the root_folder: {app.config["ROOT_FOLDER"]}\n'
                             f'Please specify a different root_folder while starting the web app.')

        for folder_name in ['ROOT_FOLDER', 'DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER']:
            os.makedirs(app.config[folder_name], exist_ok=True)


def clear_temp_folder(app: Flask) -> None:
    """Clears the temporary folder."""
    shutil.rmtree(app.config['TEMP_FOLDER'])
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
