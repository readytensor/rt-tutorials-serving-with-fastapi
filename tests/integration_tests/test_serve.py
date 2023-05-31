from fastapi.testclient import TestClient
import pytest

from serve import  create_app



@pytest.fixture
def app(model_resources):
    """Define a fixture for the test app."""
    return TestClient(create_app(model_resources))


@pytest.fixture
def sample_request_data():
    # Define a fixture for test data
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

@pytest.fixture
def sample_response_data():
    # Define a fixture for expected response
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

