.. _web:

Web Interface
=============

Overview
--------

For those less familiar with the command line, Chemprop also includes a web
interface which allows for basic training and predicting.

.. image:: _static/images/web_train.png
   :alt: Training with our web interface

.. image:: _static/images/web_predict.png
   :alt: Predicting with our web interface

You can start the web interface on your local machine in two ways. Flask is used for development mode while gunicorn is used for production mode.

Flask
-----

Run :code:`chemprop_web` (or optionally :code:`python web.py` if installed from source) and then navigate to `localhost:5000 <http://localhost:5000>`_ in a web browser.

Security defaults
-----------------

The Chemprop v1 web interface is restricted to loopback clients by default.
Checkpoint uploads are disabled because v1 ``.pt`` files are pickle-based and
must never be loaded from an untrusted source. Trusted local uploads can be
enabled with :code:`--allow_checkpoint_uploads`.

Remote serving requires :code:`--allow_remote`,
:code:`CHEMPROP_WEB_PASSWORD` of at least 16 characters, and a random
:code:`CHEMPROP_WEB_SECRET_KEY` of at least 32 bytes. Deploy it only behind an
HTTPS reverse proxy, and do not enable Flask debug mode.
Remote mode uses one authenticated data namespace and rejects checkpoint
uploads.
The configured state directory contains the checkpoint-selection database and
pickle-based model files. Remote mode therefore requires that it is owned by
the service account and private (mode ``0700``).
The default state root is :code:`~/.chemprop-web`, or
:code:`CHEMPROP_WEB_ROOT` when that environment variable is set. Mutable state
is no longer written below the source checkout. When upgrading an old checkout,
copy only trusted data/checkpoints to the new private root or pass the old
location explicitly with :code:`--root_folder` after securing it.

Training progress and prediction downloads use legacy process-local state.
Run exactly one Gunicorn worker and one thread; this interface is not a
multi-worker job service.

Do not expose a loopback/default-mode Gunicorn socket through a reverse proxy:
the proxy itself appears as a loopback client. A proxied deployment must call
:code:`build_app(allow_remote=True)`, forward HTTPS Basic authentication, and
set both security environment variables. Chemprop intentionally does not trust
:code:`X-Forwarded-For` or other forwarded-address headers.

Gunicorn
--------

Gunicorn is only available for a UNIX environment, meaning it will not work on
Windows. It is included in ``environment.yml``; installs made with package
extras can add it with:

.. code-block::

   python -m pip install -e ".[web]"

For local-only use, bind explicitly to loopback:

.. code-block::

   gunicorn --workers 1 --threads 1 --bind 127.0.0.1:5000 \
     'chemprop.web.wsgi:build_app()'

For an HTTPS reverse proxy, opt into authenticated remote mode explicitly:

.. code-block::

   install -d -m 700 "$HOME/.chemprop-web"
   CHEMPROP_WEB_PASSWORD='...' CHEMPROP_WEB_SECRET_KEY='...' \
     gunicorn --workers 1 --threads 1 --bind 127.0.0.1:5000 \
     "chemprop.web.wsgi:build_app(allow_remote=True, root_folder='$HOME/.chemprop-web')"

Generate independent random values rather than reusing another service's
password. For example, :code:`python -c "import secrets; print(secrets.token_hex(32))"`
generates a suitable value for either variable.

Never publish the first command through a proxy.

   * To run this server in the background, add the :code:`--daemon` flag.
   * Arguments including :code:`init_db` and :code:`demo` can be passed with this pattern: :code:`'chemprop.web.wsgi:build_app(init_db=True, demo=True)'`
   * See the `Gunicorn documentation <https://docs.gunicorn.org/en/stable/>`_.
