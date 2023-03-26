from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import uvicorn

from data_management.schema_provider import BinaryClassificationSchema
from model_server import ModelServer
from paths import MODEL_ARTIFACTS_PATH, SCHEMA_DIR
from data_management.data_utils import read_json_in_directory

# Create an instance of the FastAPI class
app = FastAPI()

# Load the schema file
schema_dict = read_json_in_directory(file_dir_path=SCHEMA_DIR)
data_schema = BinaryClassificationSchema(schema_dict=schema_dict)

# Load the model server
model_server = ModelServer(model_path=MODEL_ARTIFACTS_PATH, data_schema=data_schema)


@app.get("/ping")
async def ping() -> dict:
    """
    GET endpoint that returns a message indicating the service is running.
    """
    return {"message": "Pong!"}


class InferenceRequest(BaseModel):
    instances: list


@app.post("/infer", tags=["inference", "json"], response_class=JSONResponse)
async def infer(request: InferenceRequest) -> dict:
    """
    POST endpoint that takes input data as a JSON object and returns the predicted class probabilities.
    """
    # Convert the JSON object to a pandas dataframe
    data = pd.DataFrame.from_records(request.instances)
    print(f"Invoked with {data.shape[0]} records")
    predictions = model_server.predict_for_online_inferences(data)
    return {
        "status": "success",
        "message": None,
        "predictions": predictions,
    }


if __name__ == "__main__":
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
