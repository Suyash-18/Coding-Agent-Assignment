import pytest
from fastapi.testclient import TestClient

import sample_project.storage as storage
from sample_project.main import app


@pytest.fixture(autouse=True)
def clean_storage():
    storage.clear()
    yield
    storage.clear()


@pytest.fixture
def client():
    return TestClient(app)