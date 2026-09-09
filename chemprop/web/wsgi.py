"""
Runs the web interface version of Chemprop.
Designed to be used for production only, along with Gunicorn.
"""
from chemprop.web.app import app, db
from chemprop.web.utils import (
    clear_temp_folder,
    set_root_folder,
    validate_web_security_config,
    validate_web_storage_config,
)


def build_app(*args, **kwargs):
    allow_remote = kwargs.get('allow_remote', False)
    debug = kwargs.get('debug', False)
    app.config['DEBUG'] = debug
    app.config['DEMO'] = kwargs.get('demo', False)
    app.config['LOCAL_ONLY'] = not allow_remote
    app.config['ALLOW_CHECKPOINT_UPLOADS'] = kwargs.get('allow_checkpoint_uploads', False)
    app.config['SESSION_COOKIE_SECURE'] = allow_remote
    validate_web_security_config(app, allow_remote=allow_remote, debug=debug)

    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=kwargs.get('root_folder', None),
        create_folders=True
    )
    if allow_remote:
        validate_web_storage_config(app)
    clear_temp_folder(app=app)

    db.init_app(app)
    if kwargs.get('init_db', False):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    return app
