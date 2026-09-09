"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""
import os

from flask import Flask

from chemprop.web.utils import set_root_folder


app = Flask(__name__)
app.config.from_object('chemprop.web.config')
# Keep mutable databases, uploaded data, and pickle-based checkpoints outside
# the installed source tree.  A private leaf under the user's home directory
# also avoids group-writable editable checkouts on shared systems.
default_root_folder = os.path.abspath(
    os.environ.get(
        'CHEMPROP_WEB_ROOT',
        os.path.join(os.path.expanduser('~'), '.chemprop-web'),
    )
)
set_root_folder(
    app=app,
    root_folder=default_root_folder,
    create_folders=False
)

# Importing views registers the Flask routes on ``app``.
from chemprop.web.app import views  # noqa: F401
