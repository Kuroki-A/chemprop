.. _web:

Web Interface
=============

Overview
--------

For those less familiar with the command line, Chemprop also includes a web interface which allows for basic training and predicting. An example of the website (in demo mode with training disabled) is available here: `<chemprop.csail.mit.edu>`_.

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

Do not expose a loopback/default-mode Gunicorn socket through a reverse proxy:
the proxy itself appears as a loopback client. A proxied deployment must call
:code:`build_app(allow_remote=True)`, forward HTTPS Basic authentication, and
set both security environment variables. Chemprop intentionally does not trust
:code:`X-Forwarded-For` or other forwarded-address headers.

Gunicorn
--------

Gunicorn is only available for a UNIX environment, meaning it will not work on Windows. It is not installed by default with the rest of Chemprop, so first run:

.. code-block::

   pip install gunicorn

For local-only use, bind explicitly to loopback:

.. code-block::

   gunicorn --bind 127.0.0.1:5000 'chemprop.web.wsgi:build_app()'

For an HTTPS reverse proxy, opt into authenticated remote mode explicitly:

.. code-block::

   CHEMPROP_WEB_PASSWORD='...' CHEMPROP_WEB_SECRET_KEY='...' \
     gunicorn --bind 127.0.0.1:5000 \
     'chemprop.web.wsgi:build_app(allow_remote=True)'

Generate independent random values rather than reusing another service's
password. For example, :code:`python -c "import secrets; print(secrets.token_hex(32))"`
generates a suitable value for either variable.

Never publish the first command through a proxy.

   * To run this server in the background, add the :code:`--daemon` flag.
   * Arguments including :code:`init_db` and :code:`demo` can be passed with this pattern: :code:`'wsgi:build_app(init_db=True, demo=True)'`
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).
