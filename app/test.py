
import paths
from model_server import ModelServer
from data_management.schema_provider import BinaryClassificationSchema
from data_management.data_utils import read_json_in_directory, read_csv_in_directory, \
    save_csv_to_directory, validate_data
from logger import get_logger, log_error


logger = get_logger(log_file_path=paths.TEST_LOG_FILE_PATH, task_name="test")


def run_test():
    
    try:    
        # load the json file schema into a dictionary and use it to instantiate the schema provider
        logger.info("Loading schema...")
        schema_dict = read_json_in_directory(paths.SCHEMA_DIR)
        data_schema = BinaryClassificationSchema(schema_dict)

        # load test data
        logger.info("Loading test data...")
        test_data = read_csv_in_directory(file_dir_path=paths.TEST_DIR)

        # validate test data
        logger.info("Validating test data...")
        test_data = validate_data(data=test_data, data_schema=data_schema, is_train=False)

        # load model server
        logger.info("Loading model ...")
        model_server = ModelServer(model_path=paths.MODEL_ARTIFACTS_PATH, data_schema=data_schema)

        # make predictions - these are predicted class probabilities
        logger.info("Making predictions...")
        predictions = model_server.predict_proba(data=test_data)

        # save_predictions
        logger.info("Saving predictions...")
        save_csv_to_directory(
            df=predictions, 
            file_dir_path=paths.TEST_OUTPUTS_DIR,
            file_name=paths.PREDICTIONS_FILE_NAME
            )
        
        logger.info("Test predictions completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error to the general logging file 'test.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'test.error'
        log_error(message=err_msg, error=exc, error_fpath=paths.TEST_ERROR_FILE_PATH)



if __name__ == "__main__": 
    run_test()