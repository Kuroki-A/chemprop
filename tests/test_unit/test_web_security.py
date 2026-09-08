"""Security regression tests for the legacy local web interface."""

import html
import base64
from contextlib import contextmanager
from io import BytesIO
import os
import re
import sqlite3
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import zipfile

import pytest

from chemprop.web.app import app as web_app, db, views
from chemprop.web import run as web_run
from chemprop.web.wsgi import build_app


@contextmanager
def _remote_config(password='correct-horse-battery-staple', secret='s' * 32):
    """Temporarily installs explicit remote credentials on the global app."""
    keys = ('WEB_USERNAME', 'WEB_PASSWORD', 'SECRET_KEY', 'SECRET_KEY_CONFIGURED')
    previous = {key: web_app.config[key] for key in keys}
    web_app.config.update(
        WEB_USERNAME='remote',
        WEB_PASSWORD=password,
        SECRET_KEY=secret,
        SECRET_KEY_CONFIGURED=True,
    )
    try:
        yield
    finally:
        web_app.config.update(previous)


def _csrf(client):
    response = client.get('/')
    match = re.search(rb'<meta name="csrf-token" content="([^"]+)"', response.data)
    assert match is not None
    return html.unescape(match.group(1).decode())


def test_remote_clients_are_rejected_by_default():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.test_client() as client:
            response = client.get('/', environ_base={'REMOTE_ADDR': '203.0.113.7'})
            assert response.status_code == 403


def test_state_changing_routes_require_csrf():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.test_client() as client:
            assert client.post('/create_user', data={'newUserName': 'alice'}).status_code == 400
            token = _csrf(client)
            response = client.post(
                '/create_user',
                data={'newUserName': 'alice', '_csrf_token': token},
            )
            assert response.status_code == 302


def test_resource_names_reject_control_characters():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/create_user',
                data={'newUserName': 'alice\nInjected', '_csrf_token': token},
            )

        assert response.status_code == 400
        with app.app_context():
            assert len(db.get_all_users()) == 1


def test_user_selection_never_redirects_to_an_external_referrer():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.app_context():
            user_id, _ = db.insert_user('alice')

        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/select_user',
                data={'user_id': user_id, '_csrf_token': token},
                headers={'Referer': 'https://attacker.example/train?steal=1'},
            )

        assert response.status_code == 302
        assert response.headers['Location'] == '/train'


def test_upload_status_query_ignores_malformed_or_non_list_json():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.test_client() as client:
            malformed = client.get('/data?data_upload_warnings=not-json')
            wrong_type = client.get('/data?data_upload_errors=%22not-a-list%22')

        assert malformed.status_code == 200
        assert wrong_type.status_code == 200
        assert b'Warning: n' not in malformed.data
        assert b'Error: n' not in wrong_type.data


def test_dataset_delete_requires_owner_and_post():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.app_context():
            other_user_id, _ = db.insert_user('other')
            dataset_id, _ = db.insert_dataset('private', app.config['DEFAULT_USER_ID'], 'regression')

        with app.test_client() as client:
            assert client.get(f'/data/delete/{dataset_id}').status_code == 405
            token = _csrf(client)
            response = client.post(
                '/select_user',
                data={'user_id': other_user_id, '_csrf_token': token},
            )
            assert response.status_code == 302
            response = client.post(
                f'/data/delete/{dataset_id}',
                data={'_csrf_token': token},
            )
            assert response.status_code == 404

        with app.app_context():
            assert db.get_dataset(dataset_id, user_id=app.config['DEFAULT_USER_ID']) is not None


def test_checkpoint_upload_is_disabled_without_explicit_trust_opt_in():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True, ALLOW_CHECKPOINT_UPLOADS=False)
        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/checkpoints/upload/home',
                data={'_csrf_token': token},
            )
            assert response.status_code == 403


def test_checkpoint_zip_extraction_rejects_traversal_and_accepts_regular_models():
    with TemporaryDirectory() as root_dir:
        unsafe_zip = os.path.join(root_dir, 'unsafe.zip')
        destination = os.path.join(root_dir, 'models')
        os.makedirs(destination)
        with zipfile.ZipFile(unsafe_zip, 'w') as archive:
            archive.writestr('../model.pt', b'unsafe')
        with pytest.raises(ValueError, match='Unsafe checkpoint archive member'):
            views._safe_checkpoint_paths_from_zip(unsafe_zip, destination)

        safe_zip = os.path.join(root_dir, 'safe.zip')
        with zipfile.ZipFile(safe_zip, 'w') as archive:
            archive.writestr('fold_0/model.pt', b'model')
            archive.writestr('README.txt', b'ignored')
        paths = views._safe_checkpoint_paths_from_zip(safe_zip, destination)
        assert len(paths) == 1
        with open(paths[0], 'rb') as model_file:
            assert model_file.read() == b'model'


def test_checkpoint_zip_without_models_returns_an_error():
    archive_data = BytesIO()
    with zipfile.ZipFile(archive_data, 'w') as archive:
        archive.writestr('README.txt', b'not a checkpoint')
    archive_data.seek(0)

    with TemporaryDirectory() as root_dir:
        app = build_app(
            root_folder=root_dir,
            init_db=True,
            allow_checkpoint_uploads=True,
        )
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/checkpoints/upload/checkpoints',
                data={
                    'checkpointName': 'empty',
                    'checkpoint': (archive_data, 'models.zip'),
                    '_csrf_token': token,
                },
                content_type='multipart/form-data',
                follow_redirects=True,
            )

        assert response.status_code == 200
        assert b'does not contain any .pt files' in response.data


def test_database_reinitialization_respects_foreign_keys():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.app_context():
            db.insert_dataset('owned', app.config['DEFAULT_USER_ID'], 'regression')
            db.init_db()
            assert db.get_all_users()[app.config['DEFAULT_USER_ID']]['username'] == 'DEFAULT'


def test_database_insert_does_not_retry_non_unique_integrity_errors():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.app_context(), pytest.raises(sqlite3.IntegrityError, match='FOREIGN KEY'):
            db.insert_dataset('orphan', 999999, 'regression')


def test_build_app_init_db_false_preserves_existing_database():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.app_context():
            user_id, _ = db.insert_user('persistent')

        app = build_app(root_folder=root_dir, init_db=False)
        with app.app_context():
            assert user_id in db.get_all_users()


def test_build_app_is_idempotent_after_first_request_on_flask_3():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.test_client() as client:
            assert client.get('/').status_code == 200

        app = build_app(root_folder=root_dir, init_db=False)
        with app.test_client() as client:
            assert client.get('/').status_code == 200


def test_development_server_disables_concurrent_legacy_state(monkeypatch):
    run_options = {}
    monkeypatch.setattr(web_run, 'set_root_folder', lambda **_: None)
    monkeypatch.setattr(web_run, 'clear_temp_folder', lambda **_: None)
    monkeypatch.setattr(web_run, 'validate_web_security_config', lambda *_, **__: None)
    monkeypatch.setattr(web_run.db, 'init_app', lambda _: None)
    monkeypatch.setattr(web_run.os.path, 'isfile', lambda _: True)
    monkeypatch.setattr(web_run.app, 'run', lambda **kwargs: run_options.update(kwargs))

    web_run.run_web(SimpleNamespace(
        demo=False,
        allow_remote=False,
        allow_checkpoint_uploads=False,
        host='127.0.0.1',
        port=5000,
        debug=False,
        root_folder=None,
        initdb=False,
    ))

    assert run_options['threaded'] is False
    assert run_options['use_reloader'] is False


def test_remote_mode_rejects_weak_credentials_debug_and_uploads():
    with _remote_config(password='p' * 15, secret='s' * 32):
        with pytest.raises(ValueError, match='at least 16 characters'):
            build_app(allow_remote=True)

    with _remote_config(password='p' * 16, secret='s' * 31):
        with pytest.raises(ValueError, match='at least 32 bytes'):
            build_app(allow_remote=True)

    with _remote_config():
        with pytest.raises(ValueError, match='Debug mode'):
            build_app(allow_remote=True, debug=True)
        with pytest.raises(ValueError, match='Checkpoint uploads'):
            build_app(allow_remote=True, allow_checkpoint_uploads=True)


def test_remote_mode_requires_private_user_owned_storage():
    with TemporaryDirectory() as parent_dir, _remote_config():
        root_dir = os.path.join(parent_dir, 'shared-state')
        os.mkdir(root_dir, mode=0o755)

        with pytest.raises(ValueError, match='must be private'):
            build_app(root_folder=root_dir, init_db=False, allow_remote=True)


def test_security_headers_and_current_cdn_assets():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.test_client() as client:
            http_response = client.get('/')
            https_response = client.get('/', base_url='https://localhost')

        assert http_response.status_code == 200
        assert 'Strict-Transport-Security' not in http_response.headers
        assert https_response.headers['Strict-Transport-Security'].startswith('max-age=31536000')
        assert https_response.headers['X-Content-Type-Options'] == 'nosniff'
        assert https_response.headers['X-Frame-Options'] == 'DENY'
        assert https_response.headers['Referrer-Policy'] == 'no-referrer'
        assert https_response.headers['Permissions-Policy'] == 'camera=(), geolocation=(), microphone=()'
        assert https_response.headers['Cache-Control'] == 'no-store'
        assert https_response.headers['Pragma'] == 'no-cache'
        policy = https_response.headers['Content-Security-Policy']
        assert "default-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy
        assert 'https://code.jquery.com' in policy
        assert 'https://cdn.jsdelivr.net' in policy
        assert b'jquery-3.7.1.min.js' in https_response.data
        assert b'bootstrap@3.4.1' in https_response.data
        assert https_response.data.count(b'integrity=') >= 4
        assert b'ajax.googleapis.com' not in https_response.data
        assert b'maxcdn.bootstrapcdn.com' not in https_response.data


def test_flask_3_download_routes_use_supported_send_file_arguments():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.app_context():
            dataset_id, _ = db.insert_dataset('dataset', app.config['DEFAULT_USER_ID'], 'regression')
            checkpoint_id, _ = db.insert_ckpt(
                'checkpoint', app.config['DEFAULT_USER_ID'], 'regression', 1, 1, 1
            )
            model_id = db.insert_model(checkpoint_id)

        with open(os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv'), 'wb') as dataset_file:
            dataset_file.write(b'smiles,target\nCC,1\n')
        with open(os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt'), 'wb') as model_file:
            model_file.write(b'checkpoint')
        with open(os.path.join(app.config['TEMP_FOLDER'], '1_predictions.csv'), 'wb') as predictions_file:
            predictions_file.write(b'smiles,prediction\nCC,1\n')

        with app.test_client() as client:
            responses = (
                client.get('/download_predictions'),
                client.get(f'/data/download/{dataset_id}'),
                client.get(f'/checkpoints/download/{checkpoint_id}'),
            )

        assert all(response.status_code == 200 for response in responses)
        assert all('attachment' in response.headers['Content-Disposition'] for response in responses)


def test_remote_page_installs_csrf_for_ajax_requests():
    with TemporaryDirectory() as root_dir:
        with _remote_config():
            app = build_app(root_folder=root_dir, init_db=True, allow_remote=True)
            app.config.update(TESTING=False)
            authorization = base64.b64encode(b'remote:correct-horse-battery-staple').decode()
            headers = {'Authorization': f'Basic {authorization}'}
            remote = {'REMOTE_ADDR': '203.0.113.7'}
            with app.test_client() as client:
                response = client.get('/', headers=headers, environ_base=remote)
                assert response.status_code == 200
                assert response.headers['Strict-Transport-Security'].startswith('max-age=31536000')
                assert b'$.ajaxSetup' in response.data
                match = re.search(
                    rb'<meta name="csrf-token" content="([^"]+)"', response.data
                )
                assert match is not None
                token = html.unescape(match.group(1).decode())
                response = client.post(
                    '/receiver',
                    headers={**headers, 'X-CSRF-Token': token},
                    environ_base=remote,
                )
                assert response.status_code == 200

            assert stat.S_IMODE(os.stat(app.config['DATA_FOLDER']).st_mode) == 0o700
            assert stat.S_IMODE(os.stat(app.config['CHECKPOINT_FOLDER']).st_mode) == 0o700


def test_prediction_reports_mixed_invalid_smiles(monkeypatch):
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.app_context():
            checkpoint_id, _ = db.insert_ckpt(
                'checkpoint', app.config['DEFAULT_USER_ID'], 'regression', 1, 1, 2
            )
            db.insert_model(checkpoint_id)

        monkeypatch.setattr(views, 'load_task_names', lambda _: ['target'])
        monkeypatch.setattr(
            views,
            'load_args',
            lambda _: SimpleNamespace(
                features_path=None,
                features_generator=None,
                features_scaling=True,
            ),
        )
        monkeypatch.setattr(views, 'make_predictions', lambda **_: [[1.5], None])

        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/predict',
                data={
                    'checkpointName': checkpoint_id,
                    'textSmiles': 'CC invalid',
                    'drawSmiles': '',
                    '_csrf_token': token,
                },
            )

        assert response.status_code == 200
        assert b'List contains invalid SMILES strings' in response.data
        assert b'Invalid SMILES String' in response.data


def test_empty_prediction_input_is_reported_without_running_model(monkeypatch):
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.app_context():
            checkpoint_id, _ = db.insert_ckpt(
                'checkpoint', app.config['DEFAULT_USER_ID'], 'regression', 1, 1, 0
            )
            db.insert_model(checkpoint_id)

        def unexpected_prediction(**_):
            raise AssertionError('empty input must not run prediction')

        monkeypatch.setattr(views, 'make_predictions', unexpected_prediction)

        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/predict',
                data={
                    'checkpointName': checkpoint_id,
                    'textSmiles': '',
                    'drawSmiles': '',
                    'data': (BytesIO(b''), ''),
                    '_csrf_token': token,
                },
            )

        assert response.status_code == 200
        assert b'No SMILES strings given' in response.data


def test_web_rejects_checkpoint_that_requires_external_features(monkeypatch):
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True)
        with app.app_context():
            checkpoint_id, _ = db.insert_ckpt(
                'checkpoint', app.config['DEFAULT_USER_ID'], 'regression', 1, 1, 1
            )
            db.insert_model(checkpoint_id)

        monkeypatch.setattr(views, 'load_task_names', lambda _: ['target'])
        monkeypatch.setattr(
            views,
            'load_args',
            lambda _: SimpleNamespace(
                features_path=['training_features.npz'],
                features_generator=None,
                features_scaling=True,
            ),
        )

        def unexpected_prediction(**_):
            raise AssertionError('prediction with missing external features must not run')

        monkeypatch.setattr(views, 'make_predictions', unexpected_prediction)

        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/predict',
                data={
                    'checkpointName': checkpoint_id,
                    'textSmiles': 'CC',
                    'drawSmiles': '',
                    '_csrf_token': token,
                },
            )

        assert response.status_code == 200
        assert b'requires external molecular features' in response.data


def test_failed_web_training_does_not_leave_an_empty_checkpoint(monkeypatch):
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        app.config.update(TESTING=False, LOCAL_ONLY=True, PROPAGATE_EXCEPTIONS=False)
        with app.app_context():
            dataset_id, _ = db.insert_dataset(
                'training', app.config['DEFAULT_USER_ID'], 'regression'
            )
        dataset_path = os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv')
        with open(dataset_path, 'w', encoding='utf-8') as dataset_file:
            dataset_file.write('smiles,target\nCC,1.5\nCCC,2.5\n')

        def failed_training(*_, **__):
            raise RuntimeError('simulated training failure')

        monkeypatch.setattr(views, 'run_training', failed_training)

        with app.test_client() as client:
            token = _csrf(client)
            response = client.post(
                '/train',
                data={
                    'dataName': dataset_id,
                    'datasetType': 'regression',
                    'epochs': 1,
                    'ensembleSize': 1,
                    'checkpointName': 'must-not-remain',
                    'useProgressBar': 'False',
                    '_csrf_token': token,
                },
            )

        assert response.status_code == 500
        with app.app_context():
            assert db.get_ckpts(app.config['DEFAULT_USER_ID']) == []
