I have the following script to run an inference service using fastapi:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

from data_management.schema_provider import BinaryClassificationSchema
from model_server import ModelServer
from data_management.data_utils import read_json_in_directory
import paths
from logger import get_logger, log_error


logger = get_logger(log_file_path=paths.SERVE_LOG_FILE_PATH, task_name="serve")


def load_schema() -> BinaryClassificationSchema:
    """
    Load the schema file and return a BinaryClassificationSchema instance.

    Raises:
        Exception: If there is an error during schema loading.
    Returns:
        BinaryClassificationSchema: An instance of BinaryClassificationSchema.
    """
    try:
        logger.info("Loading schema...")
        schema_dict = read_json_in_directory(file_dir_path=paths.SCHEMA_DIR)
        return BinaryClassificationSchema(schema_dict=schema_dict)
    except Exception as exc:
        err_msg = "Error occurred loading schema for serving."
        # Log the error to the general logging file 'serve.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'train.error'
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        raise exc


def load_model_server(schema: BinaryClassificationSchema) -> ModelServer:
    """
    Load the model server and return a ModelServer instance.

    Args:
        schema (BinaryClassificationSchema): An instance of BinaryClassificationSchema.
    Raises:
        Exception: If there is an error during model loading.
    Returns:
        ModelServer: An instance of ModelServer.
    """
    try:
        logger.info("Loading model ...")
        return ModelServer(model_path=paths.MODEL_ARTIFACTS_PATH, data_schema=schema)
    except Exception as exc:
        err_msg = "Error occurred loading model for serving."
        # Log the error to the general logging file 'serve.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'train.error'
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        raise exc


# Load the schema file
data_schema = load_schema()

# Load the model server
model_server = load_model_server(data_schema)

# Create an instance of the FastAPI class
app = FastAPI()


@app.get("/ping")
async def ping() -> dict:
    """
    GET endpoint that returns a message indicating the service is running.

    Returns:
        dict: A dictionary with a "message" key and "Pong!" value.
    """
    return {"message": "Pong!"}


class InferenceRequest(BaseModel):
    """
    A Pydantic BaseModel for handling inference requests.

    Attributes:
        instances (list): A list of input data instances.
    """
    instances: list


def convert_preds_df_to_list_of_dicts(preds_df: pd.DataFrame) -> dict:
    """
    Convert the predictions DataFrame to a dictionary.

    Args:
        predictions (pd.DataFrame): The predictions DataFrame.
    Returns:
        dict: A dictionary with the predictions.
    """
    class_names = list(preds_df.columns)[1:]
    preds_df["__label"] = pd.DataFrame(
        preds_df[class_names], columns=class_names
    ).idxmax(axis=1)

    predictions_response = []
    for rec in preds_df.to_dict(orient="records"):
        pred_obj = {
            data_schema.id_field: rec[data_schema.id_field],
            "label": str(rec["__label"]),
            "probabilities": {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [data_schema.id_field, "__label"]
            }
        }
        predictions_response.append(pred_obj)

    return predictions_response


@app.post("/infer", tags=["inference", "json"], response_class=JSONResponse)
async def infer(request: InferenceRequest) -> dict:
    """
    POST endpoint that takes input data as a JSON object and returns the predicted class probabilities.

    Args:
        request (InferenceRequest): The InferenceRequest instance containing the input data.
    Raises:
        HTTPException: If there is an error during inference.
    Returns:
        dict: A dictionary with "status", "message", and "predictions" keys.
    """
    try:
        logger.info("Responding to inference request.")
        data = pd.DataFrame.from_records(request.dict()["instances"])
        logger.info(f"Invoked with {data.shape[0]} record(s)...")

        logger.info("Making predictions...")
        preds_df = model_server.predict_proba(data)

        logger.info("Convert predictions df to list of dictionaries...")
        preds_list = convert_preds_df_to_list_of_dicts(preds_df=preds_df)
        logger.info("Returning predictions. All done!")
        return {
            "status": "success",
            "message": None,
            "predictions": preds_list,
        }
    except Exception as exc:
        err_msg = "Error occurred during inference."
        # Log the error to the general logging file 'serve.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'train.error'
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        raise HTTPException(status_code=500, detail=f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    logger.info("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

I am using logger in this file. When I start the service by running this script, some logs are saved in the logging file. But when I make an inference request, those logs get overwritten by the logs in the `infer` endpoint function.
How do I avoid this?

My get_logger function is as follows:

```python
def get_logger(log_file_path: str, task_name: str) -> logging.Logger:
    """
    Returns a logger object with handlers to log messages to both the console and a specified log file.

    Args:
        log_file_path (str): The file path to write the log messages to.
        task_name (str): The name of the task to include in the log messages.

    Returns:
        logging.Logger: A logger object with the specified handlers.
    """
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    log_file_handler = logging.FileHandler(log_file_path, mode="w") # <--- mode="w" to overwrite the log file; "a" to append
    log_file_handler.setLevel(logging.INFO)
    log_file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_file_handler)

    return logger

```
