from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
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
        predictions = model_server.predict_for_online_inferences(data)
        logger.info("Returning predictions. All done!")
        return {
            "status": "success",
            "message": None,
            "predictions": predictions,
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
