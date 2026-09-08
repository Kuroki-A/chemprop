"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
import secrets

import torch


DEFAULT_USER_ID = 1

SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))

# The legacy web UI has no password authentication. Keep it loopback-only by
# default and require an explicit opt-in for loading externally supplied pickle-
# based PyTorch checkpoints.
LOCAL_ONLY = True
ALLOW_CHECKPOINT_UPLOADS = False
WEB_USERNAME = os.environ.get('CHEMPROP_WEB_USERNAME', 'chemprop')
WEB_PASSWORD = os.environ.get('CHEMPROP_WEB_PASSWORD')
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
SECRET_KEY_CONFIGURED = bool(os.environ.get('CHEMPROP_WEB_SECRET_KEY'))
SECRET_KEY = os.environ.get('CHEMPROP_WEB_SECRET_KEY') or secrets.token_hex(32)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
