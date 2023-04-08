import os
import numpy as np
import random
from typing import Union 
from imblearn.over_sampling import SMOTE

from data_management.schema_provider import BinaryClassificationSchema
from data_management.pipeline import get_preprocess_pipeline, save_preprocessor_and_lbl_encoder, get_label_encoder
from data_management.data_utils import read_json_in_directory, read_csv_in_directory, validate_data
from algorithm.classifier import Classifier
import paths
from logger import get_logger, log_error


logger = get_logger(log_file_path=paths.TRAIN_LOG_FILE_PATH, task_name="train")


def set_seeds(seed_value: Union[int, float]) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int or float): The seed value to use for random
            number generation. Must be an integer or a float.

    Returns:
        None
    """
    if isinstance(seed_value, (int, float)):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        logger.info(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def run_training():
    
    try:
        # set seeds 
        set_seeds(seed_value=0)     

        # load the json file schema into a dictionary and use it to instantiate the schema provider        
        logger.info("Loading schema...")
        schema_dict = read_json_in_directory(file_dir_path=paths.SCHEMA_DIR)
        data_schema = BinaryClassificationSchema(schema_dict)

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)

        # validate the data
        logger.info("Validating train data...")
        train_data = validate_data(data=train_data, data_schema=data_schema, is_train=True)        

        # create preprocessing pipeline and label encoder
        logger.info("Creating preprocessor and label encoder...")
        preprocess_pipeline = get_preprocess_pipeline(data_schema)
        label_encoder = get_label_encoder(data_schema)

        # fit preprocessing pipeline and transform data, fit label_encoder and transform labels        
        logger.info("Preprocessing data...")
        transformed_data = preprocess_pipeline.fit_transform(train_data.drop(data_schema.id_field, axis=1))
        transformed_labels = label_encoder.fit_transform(train_data[[data_schema.target_field]])

        # handle class imbalance using SMOTE       
        logger.info("Handling class imbalance ...")
        smote = SMOTE()
        balanced_data, balanced_labels = smote.fit_resample(transformed_data, transformed_labels)

        # instantiate and train classifier model 
        logger.info("Training model...")
        classifier = Classifier()
        feature_cols = [c for c in balanced_data.columns if c not in [data_schema.id_field, data_schema.target_field]]
        classifier.fit(
            train_X=balanced_data[feature_cols], 
            train_y=balanced_labels
        )

        # save preprocessor
        logger.info("Saving preprocessor and label encoder...")
        save_preprocessor_and_lbl_encoder(preprocess_pipeline, label_encoder, paths.MODEL_ARTIFACTS_PATH)

        # save model
        logger.info("Saving model...")
        classifier.save(model_path=paths.MODEL_ARTIFACTS_PATH)

        logger.info("Training completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error to the general logging file 'train.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'train.error'
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)


if __name__ == "__main__": 
    run_training()