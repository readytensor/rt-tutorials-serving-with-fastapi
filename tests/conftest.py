import pytest
import pandas as pd
import numpy as np
import json
import random
import string

from src.schema.data_schema import BinaryClassificationSchema


@pytest.fixture
def schema_dict():
    """Fixture to create a sample schema for testing"""
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "problemCategory": "binary_classification",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {
            "name": "id",
            "description": "unique identifier."
        },
        "target": {
            "name": "target_field",
            "description":  "some target desc.",
            "allowedValues" :     ["A", "B"],
            "positiveClass": "A"
        },
        "predictors": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "allowedValues": ["A", "B", "C"]
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "allowedValues": ["P", "Q", "R", "S", "T"]
            }
        ]
    }
    return valid_schema

@pytest.fixture
def schema_provider(schema_dict):
    """ Fixture to create a sample schema for testing"""
    return BinaryClassificationSchema(schema_dict)


@pytest.fixture
def model_config():
    """ Fixture to create a sample model_config json"""
    config = {
        "seed_value": 0,
        "validation_split": 0.1,        
        "prediction_field_name": "prediction"
    }
    return config


@pytest.fixture
def pipeline_config():
    """ Fixture to create a preprocessing config"""
    config = {
        "numeric_transformers": {
            "missing_indicator": {},
            "mean_median_imputer": { "imputation_method": "mean" },
            "standard_scaler": {},
            "outlier_clipper": { "min_val": -4.0, "max_val": 4.0 }
        },
        "categorical_transformers": {
            "cat_most_frequent_imputer": { "threshold": 0.1 },
            "missing_tag_imputer": {
            "imputation_method": "missing",
            "fill_value": "missing"
            },
            "rare_label_encoder": {
            "tol": 0.03,
            "n_categories": 1,
            "replace_with": "__rare__"
            },
            "one_hot_encoder": { "handle_unknown": "ignore" }
        },
        "feature_selection_preprocessing": {
            "constant_feature_dropper": { "tol": 1, "missing_values": "include" },
            "correlated_feature_dropper": {
            "threshold": 0.95
            }
        }
        }
    return config


@pytest.fixture
def pipeline_config_file_path(pipeline_config, tmpdir):
    """ Fixture to create and save a sample preprocessing_config json"""
    config_file_path = tmpdir.join('preprocessing.json')
    with open(config_file_path, 'w') as file:
        json.dump(pipeline_config, file)
    return str(config_file_path)


@pytest.fixture
def sample_data():
    """Fixture to create a larger sample DataFrame for testing"""
    np.random.seed(0)
    N = 100
    data = pd.DataFrame(
        {
            "id": range(1, N+1),
            "numeric_feature_1": np.random.randint(1, 100, size=N),
            "numeric_feature_2": np.random.normal(0, 1, size=N),
            "categorical_feature_1": np.random.choice(['A', 'B', 'C'], size=N),
            "categorical_feature_2": np.random.choice(["P", "Q", "R", "S", "T"], size=N),
            "target_field": np.random.choice(['A', 'B'], size=N)
        }
    )
    return data

@pytest.fixture
def sample_train_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_train = int(len(sample_data) * 0.8)
    return sample_data.head(N_train)


@pytest.fixture
def sample_test_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_test = int(len(sample_data) * 0.2)
    return sample_data.tail(N_test)


@pytest.fixture
def train_dir(sample_train_data, tmpdir):
    """ Fixture to create and save a sample DataFrame for testing"""
    train_data_dir = tmpdir.mkdir('train')
    train_data_file_path = train_data_dir.join('train.csv')
    sample_train_data.to_csv(train_data_file_path, index=False)
    return str(train_data_dir)

@pytest.fixture
def test_dir(sample_test_data, tmpdir):
    """ Fixture to create and save a sample DataFrame for testing"""
    test_data_dir = tmpdir.mkdir('test')
    test_data_file_path = test_data_dir.join('test.csv')
    sample_test_data.to_csv(test_data_file_path, index=False)
    return str(test_data_dir)


@pytest.fixture
def input_schema_dir(schema_dict, tmpdir):
    """ Fixture to create and save a sample schema for testing"""
    schema_dir = tmpdir.mkdir('input_schema')
    schema_file_path = schema_dir.join('schema.json')
    with open(schema_file_path, 'w') as file:
        json.dump(schema_dict, file)
    return str(schema_dir)

@pytest.fixture
def model_config_file_path(model_config, tmpdir):
    """ Fixture to create and save a sample model_config json"""
    config_file_path = tmpdir.join('model_config.json')
    with open(config_file_path, 'w') as file:
        json.dump(model_config, file)
    return str(config_file_path)


@pytest.fixture
def default_hyperparameters():
    hyperparameters = {
        "n_estimators": 200,
        "min_samples_split": 8,
        "min_samples_leaf": 4
    }
    return hyperparameters


@pytest.fixture
def default_hyperparameters_file_path(default_hyperparameters, tmpdir):
    """ Fixture to create and save a sample default_hyperparameters json"""
    config_file_path = tmpdir.join('default_hyperparameters.json')
    with open(config_file_path, 'w') as file:
        json.dump(default_hyperparameters, file)
    return str(config_file_path)



@pytest.fixture
def hpt_specs():
    config = {
        "num_trials": 5,
        "hyperparameters": [
            {
            "name": "n_estimators",
            "short_desc": "The number of trees in the forest.",
            "type": "int",
            "search_type": "uniform",
            "range_low": 50,
            "range_high": 500
            },
            {
            "name": "min_samples_split",
            "short_desc": "The minimum number of samples required to split an internal node",
            "type": "int",
            "search_type": "uniform",
            "range_low": 2,
            "range_high": 30
            },
            {
            "name": "min_samples_leaf",
            "short_desc": "The minimum number of samples required to be at a leaf node.",
            "type": "int",
            "search_type": "uniform",
            "range_low": 1,
            "range_high": 20
            }
        ]
    }
    return config


@pytest.fixture
def hpt_specs_file_path(hpt_specs, tmpdir):
    """ Fixture to create and save a sample hpt_specs json"""
    config_file_path = tmpdir.join('hpt_specs.json')
    with open(config_file_path, 'w') as file:
        json.dump(hpt_specs, file)
    return str(config_file_path)


@pytest.fixture
def predictions_df():
    """Fixture for creating a DataFrame representing predictions."""
    num_preds = 50
    # Create 5 random probabilities
    probabilities_A = [random.uniform(0, 1) for _ in range(num_preds)]
    # Subtract each probability from 1 to create a complementary probability
    probabilities_B = [1 - p for p in probabilities_A]

    # Create a DataFrame with an 'id' column and two class probability columns 'A' and 'B'
    df = pd.DataFrame({
        'id': [''.join(random.choices(string.ascii_lowercase + string.digits, k=num_preds)) 
               for _ in range(num_preds)],
        'A': probabilities_A,
        'B': probabilities_B,
    })
    return df