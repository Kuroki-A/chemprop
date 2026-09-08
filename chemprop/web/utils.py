"""Contains utility functions for the Flask web app."""

import os
import shutil
import stat
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


def validate_web_storage_config(app: Flask) -> None:
    """Requires private, user-owned state directories for remote serving.

    Chemprop v1 checkpoints are pickle-based, and the SQLite database selects
    which checkpoint will be loaded. A different local account must therefore
    not be able to replace either the files or their parent directories while
    an authenticated Web service is running.
    """
    if os.name != 'posix':
        raise ValueError('Remote Web serving is supported only on POSIX systems.')

    for config_name in ('ROOT_FOLDER', 'DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER'):
        path = os.path.abspath(app.config[config_name])
        if os.path.islink(path):
            raise ValueError(f'Remote Web storage {path} must not be a symbolic link.')

        path_stat = os.stat(path)
        if not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f'Remote Web storage {path} must be a directory.')
        if path_stat.st_uid != os.geteuid():
            raise ValueError(f'Remote Web storage {path} must be owned by the service user.')
        if stat.S_IMODE(path_stat.st_mode) & 0o077:
            raise ValueError(
                f'Remote Web storage {path} must be private (mode 0700); '
                f'found {stat.S_IMODE(path_stat.st_mode):04o}.'
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
        for folder_name in ['ROOT_FOLDER', 'DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER']:
            try:
                os.makedirs(app.config[folder_name], mode=0o700, exist_ok=True)
            except OSError as error:
                raise ValueError(
                    f'Unable to create Web storage directory '
                    f'{app.config[folder_name]}: {error}'
                ) from error


def clear_temp_folder(app: Flask) -> None:
    """Clears the temporary folder."""
    path = app.config['TEMP_FOLDER']
    if os.path.islink(path):
        raise ValueError(f'Refusing to clear symbolic-link temporary directory: {path}')
    shutil.rmtree(path)
    os.makedirs(path, mode=0o700)
