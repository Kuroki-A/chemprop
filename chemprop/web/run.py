"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

import os

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.web.app import app, db
from chemprop.web.utils import (
    clear_temp_folder,
    set_root_folder,
    validate_web_security_config,
    validate_web_storage_config,
)


class WebArgs(Tap):
    host: str = '127.0.0.1'  # Host IP address
    port: int = 5000  # Port
    debug: bool = False  # Whether to run in debug mode
    demo: bool = False  # Display only demo features
    initdb: bool = False  # Initialize Database
    root_folder: str = None  # Root folder for Web state (defaults to ~/.chemprop-web or CHEMPROP_WEB_ROOT)
    allow_remote: bool = False  # Allow non-loopback clients (requires deployment-layer authentication)
    allow_checkpoint_uploads: bool = False  # Trust and load uploaded pickle-based PyTorch checkpoints


def run_web(args: WebArgs) -> None:
    app.config['DEMO'] = args.demo
    app.config['LOCAL_ONLY'] = not args.allow_remote
    app.config['ALLOW_CHECKPOINT_UPLOADS'] = args.allow_checkpoint_uploads
    app.config['SESSION_COOKIE_SECURE'] = args.allow_remote

    if args.host not in {'127.0.0.1', 'localhost', '::1'} and not args.allow_remote:
        raise ValueError('Refusing to expose the unauthenticated legacy web UI. Pass --allow_remote only behind authentication.')
    validate_web_security_config(app, allow_remote=args.allow_remote, debug=args.debug)

    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=args.root_folder,
        create_folders=True
    )
    if args.allow_remote:
        validate_web_storage_config(app)
    clear_temp_folder(app=app)

    db.init_app(app)

    if args.initdb or not os.path.isfile(app.config['DB_PATH']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    # Training progress and per-user prediction paths are process-local legacy
    # state.  Keep the development server single-threaded as documented for
    # Gunicorn, and disable the debug reloader's second process.
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=False,
        use_reloader=False,
    )


def chemprop_web() -> None:
    """Runs the Chemprop website locally.

    This is the entry point for the command line command :code:`chemprop_web`.
    """
    run_web(args=WebArgs().parse_args())
