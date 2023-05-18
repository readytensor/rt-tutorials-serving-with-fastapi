
import argparse
from config import paths
from schema.data_schema import load_json_data_schema, save_schema
from config import paths
from utils import (
    set_seeds,
    load_and_split_data,
    read_json_as_dict
)
from preprocessing.preprocess import (
    train_pipeline_and_target_encoder,
    transform_data,
    save_pipeline_and_target_encoder,
    handle_class_imbalance
)
from prediction.predictor_model import (
    train_predictor_model,
    evaluate_predictor_model,
    save_predictor_model
)
from hyperparameter_tuning.tuner import tune_hyperparameters


def run_training(
        run_tuning: bool,
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        saved_schema_path: str = paths.SAVED_SCHEMA_PATH,
        model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
        train_dir: str = paths.TRAIN_DIR,
        pipeline_file_path: str = paths.PIPELINE_FILE_PATH,
        target_encoder_file_path: str = paths.TARGET_ENCODER_FILE_PATH,
        predictor_file_path: str = paths.PREDICTOR_FILE_PATH,
        hpt_results_file_path: str = paths.HPT_RESULTS_FILE_PATH,
        seed_value: int = 0,
        validation_split: float = 0.2) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model configuration file.
        train_dir (str, optional): The directory path of the train data.
        pipeline_file_path (str, optional): The path where to save the pipeline.
        target_encoder_file_path (str, optional): The path where to save the target encoder.
        predictor_file_path (str, optional): The path where to save the predictor model.
        hpt_results_file_path (str, optional): The path where to save the HPT results.
        seed_value (int, optional): The seed value to use for reproducibility. Default is 0.
        validation_split (float, optional): The proportion of the dataset to include in the validation split. Default is 0.2.

    Returns:
        None
    """
    set_seeds(seed_value=seed_value)

    # load and save schema
    data_schema = load_json_data_schema(input_schema_dir)
    save_schema(schema=data_schema, output_path=saved_schema_path)

    # load model config
    model_config = read_json_as_dict(model_config_file_path)

    # load train data and perform train/validation split
    train_split, val_split = load_and_split_data(
        file_dir_path=train_dir,
        val_pct=model_config.get("validation_split", validation_split))

    # fit and transform using pipeline and target encoder, then save them
    pipeline, target_encoder = train_pipeline_and_target_encoder(
        data_schema, train_split)
    transformed_train_inputs, transformed_train_targets = transform_data(
        pipeline, target_encoder, train_split)
    transformed_val_inputs, transformed_val_labels = transform_data(
        pipeline, target_encoder, val_split)
    balanced_train_inputs, balanced_train_labels = \
        handle_class_imbalance(transformed_train_inputs,
                               transformed_train_targets)
    save_pipeline_and_target_encoder(
        pipeline, target_encoder,
        pipeline_file_path,
        target_encoder_file_path)
    
    # hyperparameter tuning + training the model
    if run_tuning:
        tuned_hyperparameters = tune_hyperparameters(
            train_X=balanced_train_inputs,
            train_y=balanced_train_labels,
            valid_X=transformed_val_inputs,
            valid_y=transformed_val_labels,
            hpt_results_file_path=hpt_results_file_path,
            is_minimize=False)
        predictor = train_predictor_model(
            balanced_train_inputs,
            balanced_train_labels,
            hyperparameters=tuned_hyperparameters)
    else:
        # uses default hyperparameters to train model
        predictor = train_predictor_model(balanced_train_inputs, balanced_train_labels)

    # save predictor model
    save_predictor_model(predictor, predictor_file_path)

    # calculate and print validation accuracy
    val_accuracy = evaluate_predictor_model(
        predictor, transformed_val_inputs, transformed_val_labels)
    print("Validation accuracy:", round(val_accuracy, 3))

    print("Training completed successfully")


def parse_arguments() -> argparse.Namespace:
    """Parse the command line argument that indicates if user wants to run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument(
        "-t",
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training the model. If not set, use default hyperparameters.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_training(run_tuning=args.tune)
