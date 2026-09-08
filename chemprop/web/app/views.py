"""Defines a number of routes/views for the flask app."""

from functools import wraps
import hmac
import io
import ipaddress
import os
import secrets
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Callable, List, Tuple
import multiprocessing as mp
import zipfile
from urllib.parse import urlsplit

from flask import abort, json, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
import numpy as np
from rdkit import Chem
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from chemprop.web.app import app, db

from chemprop.args import PredictArgs, TrainArgs
from chemprop.constants import MODEL_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_header, get_smiles, get_task_names, validate_data
from chemprop.train import make_predictions, run_training
from chemprop.utils import create_logger, load_task_names, load_args

TRAINING = 0
PROGRESS = mp.Value('d', 0.0)
SAFE_RETURN_PAGES = {'home', 'train', 'predict', 'data', 'checkpoints'}
SAFE_REFERRER_ENDPOINTS = {
    '/': 'home',
    '/train': 'train',
    '/predict': 'predict',
    '/data': 'data',
    '/checkpoints': 'checkpoints',
    '/create_user': 'create_user',
}
MAX_RESOURCE_NAME_LENGTH = 255


def current_user_id() -> int:
    """Returns the current signed-session user namespace."""
    # The legacy schema has no per-user credentials. Remote deployments
    # therefore use a single authenticated namespace instead of exposing the
    # local profile switcher as an ownership bypass.
    if not app.config.get('LOCAL_ONLY', True):
        return int(app.config['DEFAULT_USER_ID'])

    user_id = session.get('current_user_id', app.config['DEFAULT_USER_ID'])
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        user_id = app.config['DEFAULT_USER_ID']

    if user_id not in db.get_all_users():
        user_id = app.config['DEFAULT_USER_ID']
    session['current_user_id'] = user_id
    return user_id


def csrf_token() -> str:
    """Returns a per-session CSRF token used by all state-changing routes."""
    token = session.get('_csrf_token')
    if token is None:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


app.jinja_env.globals['csrf_token'] = csrf_token


def _validated_resource_name(field_name: str) -> str:
    """Returns a bounded, non-empty name from a submitted form."""
    value = request.form.get(field_name, '').strip()
    if not value or len(value) > MAX_RESOURCE_NAME_LENGTH or not value.isprintable():
        abort(
            400,
            description=(
                f'{field_name} must contain 1 to {MAX_RESOURCE_NAME_LENGTH} '
                'printable characters.'
            ),
        )
    return value


def _required_int_form_value(field_name: str) -> int:
    """Parses a required integer form field or terminates with HTTP 400."""
    try:
        return int(request.form[field_name])
    except (KeyError, TypeError, ValueError):
        # Raising explicitly keeps the terminating control flow visible to
        # static analysis as well as to Flask.
        raise BadRequest(description=f'{field_name} must be an integer.') from None


def _apply_gpu_selection(args, gpu: str) -> None:
    """Validates a web GPU selection before applying it to Chemprop args."""
    if gpu is None:
        return
    if gpu == 'None':
        args.cuda = False
        return

    try:
        gpu_id = int(gpu)
    except (TypeError, ValueError):
        abort(400, description='Invalid GPU selection.')
    if gpu_id not in app.config['GPUS']:
        abort(400, description='The selected GPU is not available.')
    args.gpu = gpu_id


@app.context_processor
def inject_security_context():
    return {'current_user_id': current_user_id()}


@app.before_request
def enforce_web_security():
    """Keeps the unauthenticated legacy UI local and checks CSRF tokens."""
    if app.config.get('LOCAL_ONLY', True):
        try:
            if not ipaddress.ip_address(request.remote_addr or '127.0.0.1').is_loopback:
                abort(403, description='The Chemprop legacy web UI is configured for loopback access only.')
        except ValueError:
            abort(403)
    else:
        authorization = request.authorization
        valid_credentials = (
            authorization is not None
            and hmac.compare_digest(authorization.username or '', app.config['WEB_USERNAME'])
            and hmac.compare_digest(authorization.password or '', app.config.get('WEB_PASSWORD') or '')
        )
        if not valid_credentials:
            return 'Authentication required.', 401, {'WWW-Authenticate': 'Basic realm="Chemprop"'}

    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'} and not app.config.get('TESTING', False):
        supplied = request.form.get('_csrf_token') or request.headers.get('X-CSRF-Token')
        expected = session.get('_csrf_token')
        if expected is None or supplied is None or not hmac.compare_digest(expected, supplied):
            abort(400, description='Missing or invalid CSRF token.')


@app.after_request
def add_security_headers(response):
    """Adds browser security controls to every web response."""
    response.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; "
        "base-uri 'self'; "
        "connect-src 'self'; "
        "font-src 'self' data: https://cdn.jsdelivr.net; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "frame-src 'self'; "
        "img-src 'self' data: blob:; "
        "object-src 'none'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://code.jquery.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "worker-src 'self' blob:"
    )
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault('Permissions-Policy', 'camera=(), geolocation=(), microphone=()')
    if request.endpoint != 'static':
        response.headers.setdefault('Cache-Control', 'no-store')
        response.headers.setdefault('Pragma', 'no-cache')
    # Remote mode is contractually served through HTTPS. The backend request
    # may still be plain HTTP after TLS termination, so preserve HSTS there
    # without trusting spoofable forwarded-proto headers.
    if request.is_secure or not app.config.get('LOCAL_ONLY', True):
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response


def _safe_checkpoint_paths_from_zip(zip_path: str, destination: str) -> List[str]:
    """Extracts only regular ``.pt`` files while rejecting traversal and zip bombs."""
    max_files = 100
    max_uncompressed_size = 500 * 1024 * 1024
    paths = []
    total_size = 0

    with zipfile.ZipFile(zip_path, mode='r') as archive:
        archive_members = archive.infolist()
        if len(archive_members) > max_files:
            raise ValueError(f'Checkpoint archive contains more than {max_files} entries.')
        members = [member for member in archive_members if not member.is_dir()]
        if len(members) > max_files:
            raise ValueError(f'Checkpoint archive contains more than {max_files} files.')

        for member in members:
            normalized = os.path.normpath(member.filename)
            file_type = (member.external_attr >> 16) & 0o170000
            if normalized.startswith(('..', '/')) or file_type == 0o120000:
                raise ValueError(f'Unsafe checkpoint archive member: {member.filename!r}')
            if not normalized.lower().endswith('.pt'):
                continue

            total_size += member.file_size
            if total_size > max_uncompressed_size:
                raise ValueError('Checkpoint archive is too large after decompression.')

            output_path = os.path.join(destination, f'model_{len(paths)}.pt')
            with archive.open(member) as source, open(output_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            paths.append(output_path)

    return paths


def _predictions_path() -> str:
    return os.path.join(
        app.config['TEMP_FOLDER'],
        f'{current_user_id()}_{app.config["PREDICTIONS_FILENAME"]}',
    )


def check_not_demo(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run in DEMO mode.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on DEMO flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if app.config['DEMO']:
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function


def progress_bar(args: TrainArgs, progress: mp.Value, stop_event) -> None:
    """
    Updates a progress bar displayed during training.

    :param args: Arguments.
    :param progress: The current progress.
    """
    # no code to handle crashes in model training yet, though
    current_epoch = -1
    while current_epoch < args.epochs - 1 and not stop_event.is_set():
        if os.path.exists(os.path.join(args.save_dir, 'verbose.log')):
            with open(os.path.join(args.save_dir, 'verbose.log'), 'r') as f:
                content = f.read()
                if 'Epoch ' + str(current_epoch + 1) in content:
                    current_epoch += 1
                    progress.value = (current_epoch + 1) * 100 / args.epochs
        stop_event.wait(0.1)


def find_unused_path(path: str) -> str:
    """
    Given an initial path, finds an unused path by appending different numbers to the filename.

    :param path: An initial path.
    :return: An unused path.
    """
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)

    i = 2
    while os.path.exists(path):
        path = base_name + str(i) + ext
        i += 1

    return path


def name_already_exists_message(thing_being_named: str, original_name: str, new_name: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_name: The original name of the object.
    :param new_name: The new name of the object.
    :return: A string with a message about the changed name.
    """
    return f'{thing_being_named} "{original_name}" already exists. ' \
           f'Saving to "{new_name}".'


def get_upload_warnings_errors(upload_item: str) -> Tuple[List[str], List[str]]:
    """
    Gets any upload warnings passed along in the request.

    :param upload_item: The thing being uploaded (ex. Data, Checkpoint).
    :return: A tuple with a list of warning messages and a list of error messages.
    """
    warnings_raw = request.args.get(f'{upload_item}_upload_warnings')
    errors_raw = request.args.get(f'{upload_item}_upload_errors')
    def decode_messages(raw):
        if raw is None:
            return None
        try:
            messages = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(messages, list):
            return None
        messages = [message for message in messages if isinstance(message, str)]
        return messages[:100] or None

    warnings = decode_messages(warnings_raw)
    errors = decode_messages(errors_raw)

    return warnings, errors


def format_float(value: float, precision: int = 4) -> str:
    """
    Formats a float value to a specific precision.

    :param value: The float value to format.
    :param precision: The number of decimal places to use.
    :return: A string containing the formatted float.
    """
    return f'{value:.{precision}f}'


def format_float_list(array: List[float], precision: int = 4) -> List[str]:
    """
    Formats a list of float values to a specific precision.

    :param array: A list of float values to format.
    :param precision: The number of decimal places to use.
    :return: A list of strings containing the formatted floats.
    """
    return [format_float(f, precision) for f in array]


@app.route('/receiver', methods=['POST'])
@check_not_demo
def receiver():
    """Receiver monitoring the progress of training."""
    return jsonify(progress=PROGRESS.value, training=TRAINING)


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html', users=db.get_all_users())


@app.route('/select_user', methods=['POST'])
@check_not_demo
def select_user():
    """Selects a local user namespace in the signed Flask session."""
    if not app.config.get('LOCAL_ONLY', True):
        abort(403)
    try:
        user_id = int(request.form['user_id'])
    except (KeyError, TypeError, ValueError):
        abort(400)
    if user_id not in db.get_all_users():
        abort(404)
    session['current_user_id'] = user_id
    referrer_path = urlsplit(request.referrer or '').path
    return_endpoint = SAFE_REFERRER_ENDPOINTS.get(referrer_path, 'home')
    return redirect(url_for(return_endpoint))


@app.route('/create_user', methods=['GET', 'POST'])
@check_not_demo
def create_user():
    """
    If a POST request is made, creates a new user.
    Renders the create_user page.
    """
    if request.method == 'GET':
        return render_template('create_user.html', users=db.get_all_users())

    db.insert_user(_validated_resource_name('newUserName'))

    return redirect(url_for('create_user'))


def render_train(**kwargs):
    """Renders the train page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('train.html',
                           datasets=db.get_datasets(current_user_id()),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/train', methods=['GET', 'POST'])
@check_not_demo
def train():
    """Renders the train page and performs training if request method is POST."""
    global PROGRESS, TRAINING

    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name = _required_int_form_value('dataName')
    epochs = _required_int_form_value('epochs')
    ensemble_size = _required_int_form_value('ensembleSize')
    if epochs < 1 or ensemble_size < 1:
        abort(400, description='Epochs and ensemble size must be positive integers.')
    checkpoint_name = _validated_resource_name('checkpointName')
    gpu = request.form.get('gpu')
    dataset_row = db.get_dataset(data_name, user_id=current_user_id())
    if dataset_row is None:
        abort(404)
    data_path = os.path.join(app.config['DATA_FOLDER'], f'{data_name}.csv')
    dataset_type = request.form.get('datasetType', 'regression')
    if dataset_type not in {'classification', 'regression'}:
        abort(400, description='Unsupported dataset type.')
    use_progress_bar = request.form.get('useProgressBar', 'True') == 'True'

    # Create and modify args
    args = TrainArgs().parse_args([
        '--data_path', data_path,
        '--dataset_type', dataset_type,
        '--epochs', str(epochs),
        '--ensemble_size', str(ensemble_size),
    ])

    # Get task names
    args.task_names = get_task_names(path=data_path, smiles_columns=args.smiles_columns)

    # Check if regression/classification selection matches data
    data = get_data(path=data_path, smiles_columns=args.smiles_columns)
    # Set the number of molecules through the length of the smiles_columns for now, we need to add an option to the site later

    targets = data.targets()
    unique_targets = {target for row in targets for target in row if target is not None}

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        errors.append('Selected classification dataset but not all labels are 0 or 1. Select regression instead.')

        return render_train(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append('Selected regression dataset but all labels are 0 or 1. Select classification instead.')

        return render_train(warnings=warnings, errors=errors)

    _apply_gpu_selection(args, gpu)

    current_user = current_user_id()

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir

        progress_stop_event = None
        process = None
        if use_progress_bar:
            progress_stop_event = mp.Event()
            process = mp.Process(target=progress_bar, args=(args, PROGRESS, progress_stop_event))
            process.start()
            TRAINING = 1

        # Run training
        logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
        try:
            _, test_scores = run_training(args, data, 0, logger)
            task_scores = test_scores[args.metrics[0]]
        finally:
            if process is not None:
                progress_stop_event.set()
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

                # Reset globals even after early stopping or a training error.
                TRAINING = 0
                PROGRESS = mp.Value('d', 0.0)

        model_paths = sorted(
            os.path.join(root, filename)
            for root, _, files in os.walk(args.save_dir)
            for filename in files
            if filename.endswith('.pt')
        )
        if not model_paths:
            raise RuntimeError('Training completed without producing a model checkpoint.')

        # Only create database records after training succeeds. This avoids
        # leaving an empty checkpoint behind when training raises an error.
        ckpt_id, ckpt_name = db.insert_ckpt(checkpoint_name,
                                            current_user,
                                            args.dataset_type,
                                            args.epochs,
                                            args.ensemble_size,
                                            len(targets))

        # Check if name overlap
        if checkpoint_name != ckpt_name:
            warnings.append(name_already_exists_message('Checkpoint', checkpoint_name, ckpt_name))

        # Move models
        for model_path in model_paths:
            model_id = db.insert_model(ckpt_id)
            save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')
            shutil.move(model_path, save_path)

    return render_train(trained=True,
                        metric=args.metric,
                        num_tasks=len(args.task_names),
                        task_names=args.task_names,
                        task_scores=format_float_list(task_scores),
                        mean_score=format_float(np.mean(task_scores)),
                        warnings=warnings,
                        errors=errors)


def render_predict(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('predict.html',
                           checkpoints=db.get_ckpts(current_user_id()),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_predict()

    # Get arguments
    try:
        ckpt_id = int(request.form['checkpointName'])
    except (TypeError, ValueError):
        abort(400)

    predictions_path = _predictions_path()
    if os.path.isfile(predictions_path):
        os.remove(predictions_path)

    text_smiles = request.form.get('textSmiles', '').strip()
    draw_smiles = request.form.get('drawSmiles', '').strip()
    if text_smiles:
        smiles = text_smiles.split()
    elif draw_smiles:
        smiles = [draw_smiles]
    else:
        # Upload data file with SMILES
        data = request.files.get('data')
        if data is None or not secure_filename(data.filename or ''):
            return render_predict(errors=['No SMILES strings given'])
        with NamedTemporaryFile(suffix='.csv') as temp_file:
            data.save(temp_file.name)
            header = get_header(temp_file.name)
            possible_smiles = header[0] if header else None
            smiles = [possible_smiles] if possible_smiles and Chem.MolFromSmiles(possible_smiles) is not None else []
            smiles.extend(get_smiles(temp_file.name))

    if not smiles:
        return render_predict(errors=['No SMILES strings given'])

    smiles = [[s] for s in smiles]

    if db.get_ckpt(ckpt_id, user_id=current_user_id()) is None:
        abort(404)
    models = db.get_models(ckpt_id, user_id=current_user_id())
    if not models:
        abort(404)
    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    train_args = load_args(model_paths[0])

    # Build arguments
    arguments = [
        '--test_path', 'None',
        '--preds_path', _predictions_path(),
        '--checkpoint_paths', *model_paths
    ]

    if gpu is not None:
        if gpu == 'None':
            arguments.append('--no_cuda')
        else:
            try:
                gpu_id = int(gpu)
            except (TypeError, ValueError):
                abort(400, description='Invalid GPU selection.')
            if gpu_id not in app.config['GPUS']:
                abort(400, description='The selected GPU is not available.')
            arguments += ['--gpu', str(gpu_id)]

    # The legacy Web form cannot collect row-aligned external molecular,
    # phase, atom, bond, or constraint files. Guessing a generated descriptor
    # here previously produced dimension errors or, worse, plausible-looking
    # predictions from the wrong features. Direct these models to the CLI.
    if train_args.features_path is not None:
        return render_predict(errors=[
            'This checkpoint requires external molecular features. '
            'Use chemprop_predict with the matching --features_path file.'
        ])
    unsupported_external_inputs = (
        'phase_features_path',
        'atom_descriptors_path',
        'bond_descriptors_path',
        'constraints_path',
    )
    if any(getattr(train_args, field, None) is not None for field in unsupported_external_inputs):
        return render_predict(errors=[
            'This checkpoint requires external row-aligned inputs that the Web interface '
            'cannot collect. Use chemprop_predict with the matching input files.'
        ])
    if getattr(train_args, 'number_of_molecules', 1) != 1:
        return render_predict(errors=[
            'The Web interface supports only one molecule column per prediction. '
            'Use chemprop_predict for multi-molecule checkpoints.'
        ])

    if train_args.features_generator is not None:
        arguments += ['--features_generator', *train_args.features_generator]

        if not train_args.features_scaling:
            arguments.append('--no_features_scaling')

    # Parse arguments
    args = PredictArgs().parse_args(arguments)

    # Run predictions
    preds = make_predictions(args=args, smiles=smiles, return_uncertainty=False)

    if not preds:
        return render_predict(errors=['No SMILES strings given'])
    invalid_smiles_count = sum(pred is None for pred in preds)
    if invalid_smiles_count == len(preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = 'Invalid SMILES String'
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          warnings=["List contains invalid SMILES strings"] if invalid_smiles_count else None,
                          errors=None)


@app.route('/download_predictions')
def download_predictions():
    """Downloads predictions as a .csv file."""
    predictions_path = _predictions_path()
    if not os.path.isfile(predictions_path):
        abort(404)
    return send_from_directory(
        app.config['TEMP_FOLDER'],
        os.path.basename(predictions_path),
        as_attachment=True,
        max_age=0,
    )


@app.route('/data')
@check_not_demo
def data():
    """Renders the data page."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('data.html',
                           datasets=db.get_datasets(current_user_id()),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users())


@app.route('/data/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_data(return_page: str):
    """
    Uploads a data .csv file.

    :param return_page: The name of the page to render to after uploading the dataset.
    """
    warnings, errors = [], []

    if return_page not in SAFE_RETURN_PAGES:
        abort(400)
    current_user = current_user_id()

    dataset = request.files['dataset']

    with NamedTemporaryFile() as temp_file:
        dataset.save(temp_file.name)
        dataset_errors = validate_data(temp_file.name)

        if len(dataset_errors) > 0:
            errors.extend(dataset_errors)
        else:
            dataset_name = _validated_resource_name('datasetName')
            # dataset_class = load_args(ckpt).dataset_type  # TODO: SWITCH TO ACTUALLY FINDING THE CLASS

            dataset_id, new_dataset_name = db.insert_dataset(dataset_name, current_user, 'UNKNOWN')

            dataset_path = os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv')

            if dataset_name != new_dataset_name:
                warnings.append(name_already_exists_message('Data', dataset_name, new_dataset_name))

            shutil.copy(temp_file.name, dataset_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


@app.route('/data/download/<int:dataset>')
@check_not_demo
def download_data(dataset: int):
    """
    Downloads a dataset as a .csv file.

    :param dataset: The id of the dataset to download.
    """
    if db.get_dataset(dataset, user_id=current_user_id()) is None:
        abort(404)
    return send_from_directory(
        app.config['DATA_FOLDER'], f'{dataset}.csv', as_attachment=True, max_age=0
    )


@app.route('/data/delete/<int:dataset>', methods=['POST'])
@check_not_demo
def delete_data(dataset: int):
    """
    Deletes a dataset.

    :param dataset: The id of the dataset to delete.
    """
    if not db.delete_dataset(dataset, user_id=current_user_id()):
        abort(404)
    path = os.path.join(app.config['DATA_FOLDER'], f'{dataset}.csv')
    if os.path.isfile(path):
        os.remove(path)
    return redirect(url_for('data'))


@app.route('/checkpoints')
@check_not_demo
def checkpoints():
    """Renders the checkpoints page."""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('checkpoints.html',
                           checkpoints=db.get_ckpts(current_user_id()),
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_checkpoint(return_page: str):
    """
    Uploads a checkpoint .pt file.

    :param return_page: The name of the page to render after uploading the checkpoint file.
    """
    warnings, errors = [], []

    if not app.config.get('ALLOW_CHECKPOINT_UPLOADS', False):
        abort(403, description='Checkpoint upload is disabled because PyTorch v1 checkpoints use unsafe pickle loading.')
    if return_page not in SAFE_RETURN_PAGES:
        abort(400)
    current_user = current_user_id()

    ckpt = request.files['checkpoint']

    ckpt_name = _validated_resource_name('checkpointName')
    ckpt_ext = os.path.splitext(secure_filename(ckpt.filename))[1].lower()

    # Collect paths to all uploaded checkpoints (and unzip if necessary)
    temp_dir = TemporaryDirectory()
    ckpt_paths = []

    if ckpt_ext == '.pt':
        ckpt_path = os.path.join(temp_dir.name, MODEL_FILE_NAME)
        ckpt.save(ckpt_path)
        ckpt_paths = [ckpt_path]

    elif ckpt_ext == '.zip':
        ckpt_dir = os.path.join(temp_dir.name, 'models')
        os.makedirs(ckpt_dir)
        zip_path = os.path.join(temp_dir.name, 'models.zip')
        ckpt.save(zip_path)
        try:
            ckpt_paths = _safe_checkpoint_paths_from_zip(zip_path, ckpt_dir)
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            errors.append(str(error))
        if not ckpt_paths and not errors:
            errors.append('Uploaded checkpoint archive does not contain any .pt files.')

    else:
        errors.append(f'Uploaded checkpoint(s) file must be either .pt or .zip but got {ckpt_ext}')

    # Insert checkpoints into database
    if len(ckpt_paths) > 0 and not errors:
        try:
            # This is intentionally reachable only after the explicit
            # allow_checkpoint_uploads trust opt-in above.
            ckpt_args = load_args(ckpt_paths[0])
        except Exception as error:
            errors.append(f'Unable to load trusted checkpoint metadata: {error}')

    if len(ckpt_paths) > 0 and not errors:
        ckpt_id, new_ckpt_name = db.insert_ckpt(ckpt_name,
                                                current_user,
                                                ckpt_args.dataset_type,
                                                ckpt_args.epochs,
                                                len(ckpt_paths),
                                                ckpt_args.train_data_size)

        for ckpt_path in ckpt_paths:
            model_id = db.insert_model(ckpt_id)
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')

            if ckpt_name != new_ckpt_name:
                warnings.append(name_already_exists_message('Checkpoint', ckpt_name, new_ckpt_name))

            shutil.copy(ckpt_path, model_path)

    temp_dir.cleanup()

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, checkpoint_upload_warnings=warnings, checkpoint_upload_errors=errors))


@app.route('/checkpoints/download/<int:checkpoint>')
@check_not_demo
def download_checkpoint(checkpoint: int):
    """
    Downloads a zip of model .pt files.

    :param checkpoint: The name of the checkpoint to download.
    """
    ckpt = db.get_ckpt(checkpoint, user_id=current_user_id())
    if ckpt is None:
        abort(404)
    models = db.get_models(checkpoint, user_id=current_user_id())

    model_data = io.BytesIO()

    with zipfile.ZipFile(model_data, mode='w') as z:
        for model in models:
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
            z.write(model_path, os.path.basename(model_path))

    model_data.seek(0)

    return send_file(
        model_data,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{ckpt["ckpt_name"]}.zip',
        max_age=0,
    )


@app.route('/checkpoints/delete/<int:checkpoint>', methods=['POST'])
@check_not_demo
def delete_checkpoint(checkpoint: int):
    """
    Deletes a checkpoint file.

    :param checkpoint: The id of the checkpoint to delete.
    """
    if not db.delete_ckpt(checkpoint, user_id=current_user_id()):
        abort(404)
    return redirect(url_for('checkpoints'))
