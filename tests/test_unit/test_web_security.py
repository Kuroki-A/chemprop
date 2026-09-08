"""Security regression tests for the legacy local web interface."""

import html
import base64
from contextlib import contextmanager
import os
import re
from tempfile import TemporaryDirectory

import pytest

from chemprop.web.app import app as web_app, db
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


def test_database_reinitialization_respects_foreign_keys():
    with TemporaryDirectory() as root_dir:
        app = build_app(root_folder=root_dir, init_db=True)
        with app.app_context():
            db.insert_dataset('owned', app.config['DEFAULT_USER_ID'], 'regression')
            db.init_db()
            assert db.get_all_users()[app.config['DEFAULT_USER_ID']]['username'] == 'DEFAULT'


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
