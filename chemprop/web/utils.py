"""Contains utility functions for the Flask web app."""

import errno
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


def _validate_path_rename_protection(path: str) -> None:
    """Rejects directory chains another local account could replace.

    Checking ``realpath`` and then calling ``rmtree`` is otherwise vulnerable
    to an ancestor being renamed and replaced between those operations.  A
    non-owner cannot replace a child in a non-writable parent; a sticky
    writable parent (for example ``/tmp``) is also safe when the child belongs
    to the service user.
    """
    if os.name != 'posix':
        return

    service_uid = os.geteuid()
    # Container/user-namespace mounts can map filesystem root to a nonzero UID
    # (for example ``nobody``).  Its owner is still the platform administrator
    # for this path hierarchy and must be treated like UID 0.
    filesystem_root_uid = os.stat(
        os.path.sep, follow_symlinks=False,
    ).st_uid
    trusted_ancestor_uids = {0, service_uid, filesystem_root_uid}
    child = os.path.abspath(path)
    while child != os.path.sep:
        parent = os.path.dirname(child)
        try:
            child_stat = os.stat(child, follow_symlinks=False)
            parent_stat = os.stat(parent, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f'Unable to validate Web storage path {child}: {error}'
            ) from error
        if not stat.S_ISDIR(child_stat.st_mode):
            raise ValueError(f'Web storage path component {child} must be a directory.')
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(f'Web storage path component {parent} must be a directory.')

        if parent_stat.st_uid not in trusted_ancestor_uids:
            raise ValueError(
                f'Web storage ancestor {parent} must be owned by root or the '
                'service user so its permissions cannot change after validation.'
            )
        parent_is_writable = bool(stat.S_IMODE(parent_stat.st_mode) & 0o022)
        sticky_child_is_protected = (
            bool(parent_stat.st_mode & stat.S_ISVTX)
            and child_stat.st_uid == service_uid
            and parent_stat.st_uid in trusted_ancestor_uids
        )
        if parent_is_writable and not sticky_child_is_protected:
            raise ValueError(
                f'Web storage ancestor {parent} is writable by another local '
                'account and could replace a validated path component. Remove '
                'group/other write permissions or use a sticky directory with '
                'service-owned children.'
            )
        child = parent


def _open_directory_without_symlinks(path: str) -> int:
    """Opens an absolute directory path one component at a time."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, 'O_CLOEXEC', 0)
    directory_fd = os.open(os.path.sep, flags)
    try:
        for component in os.path.abspath(path).split(os.path.sep):
            if not component:
                continue
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _clear_open_directory(directory_fd: int) -> None:
    """Recursively unlinks directory contents without following symlinks."""
    for name in os.listdir(directory_fd):
        try:
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, 'O_CLOEXEC', 0),
                dir_fd=directory_fd,
            )
        except OSError as error:
            if error.errno not in {errno.ENOTDIR, errno.ELOOP}:
                raise
            # This removes a regular file or the link itself, never its target.
            os.unlink(name, dir_fd=directory_fd)
            continue

        try:
            _clear_open_directory(child_fd)
        finally:
            os.close(child_fd)
        os.rmdir(name, dir_fd=directory_fd)


def validate_web_storage_layout(app: Flask) -> None:
    """Validates storage containment and rejects symbolic-link traversal.

    This check is required even for loopback serving because ``TEMP_FOLDER``
    is recursively cleared during startup.
    """
    root = os.path.abspath(app.config['ROOT_FOLDER'])
    if os.path.realpath(root) != root:
        raise ValueError(
            f'Web storage root {root} must not contain symbolic-link components.'
        )
    if not os.path.isdir(root):
        raise ValueError(f'Web storage root {root} must be a directory.')

    for config_name in ('DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER'):
        path = os.path.abspath(app.config[config_name])
        try:
            contained = os.path.commonpath((root, path)) == root
        except ValueError:
            contained = False
        if not contained:
            raise ValueError(f'Web storage {path} must be contained in {root}.')
        if os.path.realpath(path) != path:
            raise ValueError(
                f'Web storage {path} must not contain symbolic-link components.'
            )
        if not os.path.isdir(path):
            raise ValueError(f'Web storage {path} must be a directory.')

    database_path = os.path.abspath(app.config['DB_PATH'])
    try:
        database_contained = os.path.commonpath((root, database_path)) == root
    except ValueError:
        database_contained = False
    if (
        not database_contained
        or os.path.realpath(database_path) != database_path
        or os.path.realpath(os.path.dirname(database_path))
        != os.path.dirname(database_path)
    ):
        raise ValueError('The Web database path must remain inside the storage root.')


def validate_web_storage_config(app: Flask) -> None:
    """Requires private, user-owned state directories for remote serving.

    Chemprop v1 checkpoints are pickle-based, and the SQLite database selects
    which checkpoint will be loaded. A different local account must therefore
    not be able to replace either the files or their parent directories while
    an authenticated Web service is running.
    """
    if os.name != 'posix':
        raise ValueError('Remote Web serving is supported only on POSIX systems.')

    validate_web_storage_layout(app)

    # The root itself is private below, so protecting its directory entry also
    # protects every child between validation and startup cleanup.
    _validate_path_rename_protection(os.path.abspath(app.config['ROOT_FOLDER']))

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
        app.config['ROOT_FOLDER'] = os.path.abspath(os.path.expanduser(root_folder))
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


def clear_web_storage_folder(app: Flask, config_name: str) -> None:
    """Safely clears one managed Web directory while retaining its inode."""
    if config_name not in {'DATA_FOLDER', 'CHECKPOINT_FOLDER', 'TEMP_FOLDER'}:
        raise ValueError(f'Unsupported Web storage folder {config_name!r}.')
    validate_web_storage_layout(app)
    _validate_path_rename_protection(os.path.abspath(app.config['ROOT_FOLDER']))
    path = os.path.abspath(app.config[config_name])
    if (
        os.name == 'posix'
        and hasattr(os, 'O_DIRECTORY')
        and hasattr(os, 'O_NOFOLLOW')
    ):
        # Holding file descriptors for every traversed component makes an
        # ancestor rename harmless, while O_NOFOLLOW and fd-relative removal
        # prevent the recursive cleanup from escaping through symlinks.
        directory_fd = _open_directory_without_symlinks(path)
        try:
            _clear_open_directory(directory_fd)
            os.fchmod(directory_fd, 0o700)
        finally:
            os.close(directory_fd)
    else:
        # Remote serving is rejected on non-POSIX systems.  This fallback keeps
        # the legacy loopback-only behavior after the static layout checks.
        shutil.rmtree(path)
        os.makedirs(path, mode=0o700)


def clear_temp_folder(app: Flask) -> None:
    """Clears the temporary folder."""
    clear_web_storage_folder(app, 'TEMP_FOLDER')
