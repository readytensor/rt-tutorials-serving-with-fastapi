## Introduction

This repository demonstrates how to create an inference service for a Random Forest binary classifier model using FastAPI and uvicorn.

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users. It is referenced in the tutorial called **Serving an ML model using FastAPI**. The purpose of the tutorial series is to help AI developers create adaptable algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

## Repository Contents

The `app/` folder in the repository contains the following key folders/sub-folders:

- `data_management/` will all files related to handling and preprocessing data.
- `data_model/` contains the data model definition for the schema and data files.
- `inputs/` contains the input files related to the _titanic_ dataset.
- `model/` is a folder to save model artifacts and other assets specific to the trained model. Within this folder:
  - `artifacts/` is location to save model artifacts (i.e. the saved model including the trained preprocessing pipeline)
- `outputs/` is used to contain the predictions or other results files. When the `test.py` script is run, a predictions file called `predictions.csv` is saved in `outputs/testing_outputs/` sub-directory. Logs and errors are saved in the `outputs/logs/` sub-directory.
- The `logger.py` script is used to define the logger object that is used to log errors and other information. There is also a `log_error()` function that is used to log errors to a separate file for convenient access to criticial errors.
- The `train.py` script is used to train the model and save the model artifacts. The model artifacts are saved in the path `./app/outputs/artifacts/`. Logs and errors generated during training are saved in the path `./app/outputs/logs/` in files called `train.log` and `train.error`, respectively.
- The `test.py` script is used to run test predictions using the trained model. The script loads the model artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./app/outputs/testing_outputs/`. Logs and errors generated during testing are saved in the path `./app/outputs/logs/` in files called `test.log` and `test.error`, respectively.
- The `serve.py` script is used to serve the model as a REST API using FastAPI and uvicorn. The script loads the model artifacts and creates the API endpoints. The app listens on a GET endpoint `/ping` and a POST endpoint `/infer`. Logs and errors generated during serving are saved in the path `./app/outputs/logs/` in files called `serve.log` and `serve.error`, respectively.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the following 3 input files in the sub-directories in `./app/inputs/`:
  - Train data, which must be a CSV file, to be placed in `./app/inputs/data/training/`. File name can be any; extension must be ".csv".
  - Test data, which must be a CSV file, to be placed in `./app/inputs/data/testing/`. File name can be any; extension must be ".csv".
  - The schema file in JSON format , to be placed in `./app/inputs/data_config/`. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category. File name can be any; extension must be ".json".
- Run the script `train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./app/outputs/artifacts/`.
- Run the script `test.py` to run test predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./app/outputs/testing_outputs/`.
- Run the script `serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints.
- Send a POST request to the endpoint `/infer` using curl. See sample curl command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "instances": [
    {
      "PassengerId": "879",
      "Pclass": 3,
      "Name": "Laleff, Mr. Kristo",
      "Sex": "male",
      "Age": null,
      "SibSp": 0,
      "Parch": 0,
      "Ticket": "349217",
      "Fare": 7.8958,
      "Cabin": null,
      "Embarked": "S"
    }
  ]
}' http://localhost:8080/infer
```

The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:

```json
{
  "status": "success",
  "message": null,
  "predictions": [
    {
      "id": "879",
      "label": "0",
      "probabilities": {
        "0": 0.9975,
        "1": 0.0025
      }
    }
  ]
}
```

## OpenAPI

Since the service is implemented using FastAPI, we get automatic documentation of the APIs offered by the service. Visit the docs at `http://localhost:8080/docs`.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
fastapi==0.70.0
uvicorn==0.15.0
pydantic==1.8.2
pandas==1.5.2
numpy==1.20.3
scikit-learn==1.0
feature-engine==1.1.1
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```
