from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import json
import pytest
import os 

from serve import  create_app
from serve_utils import (
    get_model_resources
)


@pytest.fixture
def resources_paths():
    """Define a fixture for the paths to the test model resources."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_resources_path = os.path.join(cur_dir, "test_resources")
    # base_path = 'tests/test_resources'  # Replace this with the actual path
    return {
        'saved_schema_path': os.path.join(test_resources_path, 'schema.joblib'),
        'predictor_file_path': os.path.join(test_resources_path, 'predictor.joblib'),
        'pipeline_file_path': os.path.join(test_resources_path, 'pipeline.joblib'),
        'target_encoder_file_path': os.path.join(test_resources_path, 'target_encoder.joblib'),
        'model_config_file_path': os.path.join(test_resources_path, 'model_config.json')
    }


@pytest.fixture
def model_resources(resources_paths):
    """Define a fixture for the test ModelResources object."""
    return get_model_resources(**resources_paths)


@pytest.fixture
def app(model_resources):
    """Define a fixture for the test app."""
    return TestClient(create_app(model_resources))


# Define a fixture for test data
@pytest.fixture
def sample_request_data():
    return {
        "instances": [
            {
                "PassengerId": "879",
                "Pclass": 3,
                "Name": "Laleff, Mr. Kristo",
                "Sex": "male",
                "Age": None,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "349217",
                "Fare": 7.8958,
                "Cabin": None,
                "Embarked": "S"
            }
        ]
    }

# Define a fixture for expected response
@pytest.fixture
def sample_response_data():
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": [
            "0",
            "1"
        ],
        "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [
                    0.97548,
                    0.02452
                ]
            }
        ]
    }


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch('serve.transform_req_data_and_make_predictions')
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app
    The function creates a mock request and sets the expected return value of the mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with the correct arguments.
    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {"status": "success", "predictions": []}

    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response)
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the correct arguments
    mock_transform_and_predict.assert_called()


def test_infer_endpoint_integration(app, sample_request_data, sample_response_data):
    """
    End-to-end integration test for the /infer endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the /infer endpoint,
    and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the test_model_resources fixture,
    and the application's dependency on ModelResources is overridden to use this instance for the test.

    The function sends a POST request to the "/infer" endpoint with the test_sample_request_data
    using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and compares specific
    values in the returned response_data with the sample_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        app (TestClient): The test app.
        sample_request_data (dict): The fixture for test request data.
        sample_response_data (dict): The fixture for expected response data.
    Returns:
        None
    """
    response = app.post("/infer", json=sample_request_data)
    response_data = response.json()

    # assertions
    assert set(response_data.keys()) == set(response.json().keys())
    assert response_data["predictions"][0]["sampleId"] == \
        sample_response_data["predictions"][0]["sampleId"]
    assert response_data["predictions"][0]["predictedClass"] == \
        sample_response_data["predictions"][0]["predictedClass"]
