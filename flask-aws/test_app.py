import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_hello_world(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'Hello, World!'
