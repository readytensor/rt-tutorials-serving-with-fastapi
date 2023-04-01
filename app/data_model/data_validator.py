import pandas as pd
from pydantic import BaseModel, validator


def get_data_validator(schema, is_train) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    Args:
        schema (BinaryClassificationSchema): An instance of BinaryClassificationSchema.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator('data')
        def validate_data(cls, data):

            if schema.id_field not in data.columns:
                raise ValueError(f"ID field '{schema.id_field}' is not present in the given data")

            if is_train and schema.target_field not in data.columns:
                raise ValueError(f"Target field '{schema.target_field}' is not present in the given data")

            for feature in schema.features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' is not present in the given data")

            return data

    return DataValidator


