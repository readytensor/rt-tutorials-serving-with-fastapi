I need to write unit tests for the inference service I have created for my classifier model.
Here is the code:

```python
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Any
import pandas as pd
import uuid
import uvicorn

from schema.data_schema import load_saved_schema
from config import paths
from preprocessing.preprocess import (
    load_pipeline_and_target_encoder,
    transform_data
)
from prediction.predictor_model import load_predictor_model
from  predict import get_model_predictions, add_ids_to_predictions
from utils import read_json_as_dict


class DataModel:
    def __init__(self):
        self.data_schema = load_saved_schema(paths.SAVED_SCHEMA_PATH)
        self.predictor_model = load_predictor_model(paths.PREDICTOR_FILE_PATH)
        self.preprocessor, self.target_encoder = load_pipeline_and_target_encoder(
            paths.PIPELINE_FILE_PATH,
            paths.TARGET_ENCODER_FILE_PATH
        )
        self.model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)

# Create an instance of the FastAPI class
app = FastAPI()

def get_model():
    """Returns an instance of DataModel."""
    return DataModel()


@app.get("/ping")
async def ping() -> dict:
    """GET endpoint that returns a message indicating the service is running.

    Returns:
        dict: A dictionary with a "message" key and "Pong!" value.
    """
    return {"message": "Pong!"}


def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


def create_sample_prediction(sample: dict, id_field: str, class_names: List[str]) -> dict:
    """
    Create a dictionary with the prediction results for a single sample.

    Args:
        sample (dict): A single sample's prediction results.
        id_field (str): The name of the field containing the sample ID.
        class_names (list): The names of the target classes.

    Returns:
        dict: A dictionary containing the prediction results for the sample.
    """
    return {
        "sampleId": sample[id_field],
        "predictedClass": str(sample["__predicted_class"]),
        "predictedProbabilities": [
            round(sample[class_names[0]], 5),
            round(sample[class_names[1]], 5)
        ]
    }


def create_predictions_response(
        predictions_df: pd.DataFrame,
        data_schema: Any
        ) -> None:
    """
    Convert the predictions DataFrame to a response dictionary in required format.

    Args:
        transformed_data (pd.DataFrame): The transfomed input data for prediction.
        data_schema (Any): An instance of the BinaryClassificationSchema.

    Returns:
        dict: The response data in a dictionary.
    """
    class_names = data_schema.allowed_target_values
    # find predicted class which has the highest probability
    predictions_df["__predicted_class"] = predictions_df[class_names].idxmax(axis=1)
    sample_predictions = [
        create_sample_prediction(sample, data_schema.id, class_names)
        for sample in predictions_df.to_dict(orient="records")
    ]
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": pd.Timestamp.now().isoformat(),
        "requestId": generate_unique_request_id(),
        "targetClasses": class_names,
        "targetDescription": data_schema.target_description,
        "predictions": sample_predictions,
    }

    return predictions_response


class InferenceRequestBodyModel(BaseModel):
    """
    A Pydantic BaseModel for handling inference requests.

    Attributes:
        instances (list): A list of input data instances.
    """
    instances: List[dict]


async def transform_req_data_and_make_predictions(
    request: InferenceRequestBodyModel,
    model: DataModel
) -> Tuple[pd.DataFrame, dict]:
    """Transform request data and generate predictions based on request.

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model (DataModel): The data model instance.

    Returns:
        Tuple[pd.DataFrame, dict]: Tuple containing transformed data and prediction response.
    """
    data = pd.DataFrame.from_records(request.dict()["instances"])
    print(f"Predictions requested for {len(data)} samples.")
    transformed_data, _ = transform_data(model.preprocessor, model.target_encoder, data)
    predictions_df = get_model_predictions(
        transformed_data,
        model.predictor_model,
        model.data_schema.allowed_target_values,
        model.model_config["prediction_field_name"],
        return_probs=True)
    predictions_df_with_ids = add_ids_to_predictions(
        data, predictions_df, model.data_schema.id)
    predictions_response = create_predictions_response(
        predictions_df_with_ids,
        model.data_schema
    )
    return transformed_data, predictions_response


@app.post("/infer", tags=["inference"], response_class=JSONResponse)
async def infer(request: InferenceRequestBodyModel,
                model: DataModel = Depends(get_model)) -> dict:
    """POST endpoint that takes input data as a JSON object and returns
       predicted class probabilities.

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model (DataModel, optional): The data model instance. Defaults to Depends(get_model).

    Raises:
        HTTPException: If there is an error during inference.

    Returns:
        dict: A dictionary with "status", "message", and "predictions" keys.
    """
    _, predictions_response = await transform_req_data_and_make_predictions(request, model)
    return predictions_response


if __name__ == "__main__":
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Can you please help me with this?

#############################################

I would like to create an end-to-end integration test on the `infer` endpoint of my service. Note that I updated the ModelResources class so that it now takes the paths to the model files as arguments. Here is the code:

```python
class ModelResources:
    def __init__(
            self,
            saved_schema_path: str = paths.SAVED_SCHEMA_PATH,
            predictor_file_path: str = paths.PREDICTOR_FILE_PATH,
            pipeline_file_path: str = paths.PIPELINE_FILE_PATH,
            target_encoder_file_path: str = paths.TARGET_ENCODER_FILE_PATH,
            model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
        ):
        self.data_schema = load_saved_schema(saved_schema_path)
        self.predictor_model = load_predictor_model(predictor_file_path)
        self.preprocessor, self.target_encoder = load_pipeline_and_target_encoder(
            pipeline_file_path,
            target_encoder_file_path
        )
        self.model_config = read_json_as_dict(model_config_file_path)
```

Now, we can pass in the paths to the model files for testing purposes.

I want to test the `/infer` endpoint with these artifacts, and make sure the response is as expected.

I have the following questions:

1. Is such an integration test a good idea?
2. If yes, should I add this test in the same script as the other unit tests? If i create a separate script, it would mess up the directory structure of the project since tests is supposed to mirror the src directory.
3. Where would I place the test artifacts (i.e. the predictor model, preprocessing pipeline, etc.). We will likely use these for other tests as well.
