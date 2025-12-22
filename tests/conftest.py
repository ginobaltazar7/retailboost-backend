"""
Pytest configuration and shared fixtures for RetailBoost tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'air_store_id': ['store_001', 'store_002', 'store_003'],
        'visit_date': pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-03']),
        'visitors': [10, 20, 30]
    })


@pytest.fixture
def sample_datetime():
    """Create a sample datetime for testing"""
    return datetime(2025, 12, 20, 13, 15, 8)


@pytest.fixture
def sample_date():
    """Create a sample date for testing"""
    return date(2025, 12, 20)


@pytest.fixture
def sample_reservation_data():
    """Create sample reservation data"""
    return pd.DataFrame({
        'air_store_id': ['store_001', 'store_001', 'store_002'],
        'visit_datetime': pd.to_datetime(['2017-01-15 18:00:00', '2017-01-16 19:00:00', '2017-01-15 20:00:00']),
        'reserve_datetime': pd.to_datetime(['2017-01-10 10:00:00', '2017-01-14 11:00:00', '2017-01-13 12:00:00']),
        'reserve_visitors': [4, 2, 6]
    })


@pytest.fixture
def sample_store_info():
    """Create sample store info data"""
    return pd.DataFrame({
        'air_store_id': ['store_001', 'store_002'],
        'air_genre_name': ['Italian', 'Japanese'],
        'air_area_name': ['Tokyo', 'Osaka'],
        'latitude': [35.6762, 34.6937],
        'longitude': [139.6503, 135.5023]
    })


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient"""
    # Mock the database initialization to avoid requiring actual DB
    with patch('app.startup.init_task_status_table'):
        from main import app
        return TestClient(app)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data as string"""
    return """air_store_id,visit_date,visitors
store_001,2017-01-01,10
store_002,2017-01-01,20
store_003,2017-01-01,30"""


@pytest.fixture
def sample_kernel_matrix():
    """Sample kernel matrix for testing"""
    return np.random.rand(10, 10)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
