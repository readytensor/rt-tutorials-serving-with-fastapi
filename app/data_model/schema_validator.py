from typing import List, Dict
from pydantic import BaseModel, validator

class PredictorField(BaseModel):
    fieldName: str
    dataType: str

class InputDataset(BaseModel):
    idField: str
    targetField: str
    targetClass: int
    predictorFields: List[PredictorField]

class SchemaModel(BaseModel):
    problemCategory: str 
    version: str 
    inputDatasets: Dict[str, InputDataset]

    @validator('problemCategory')
    def validate_problem_category(cls, v):
        if v != "binary_classification_base":
            raise ValueError('problemCategory must be binary_classification_base')
        return v

    @validator('version')
    def validate_version(cls, v):
        if v != "1.0":
            raise ValueError('version must be 1.0')
        return v

    @validator('inputDatasets')
    def validate_input_datasets(cls, v):
        if not v:
            raise ValueError('inputDatasets must not be empty')
        if len(v) != 1 or 'binaryClassificationBaseMainInput' not in v:
            raise ValueError('inputDatasets must have a single key binaryClassificationBaseMainInput')
        input_dataset = v['binaryClassificationBaseMainInput']
        if not input_dataset.idField:
            raise ValueError('idField must be specified')
        if not input_dataset.targetField:
            raise ValueError('targetField must be specified')
        if not input_dataset.targetClass:
            raise ValueError('targetClass must be specified')
        if not input_dataset.predictorFields:
            raise ValueError('predictorFields must not be empty')
        for field in input_dataset.predictorFields:
            if not field.fieldName:
                raise ValueError('predictorFields must contain a fieldName')
            if field.dataType not in ["CATEGORICAL", "INT", "REAL", "NUMERIC"]:
                raise ValueError('Invalid dataType for predictorFields')
        return v


if __name__ == "__main__":
    # test data for the schema
    sample_schema_dict = {
        "problemCategory": "binary_classification_base",
        "version": "1.0",
        "inputDatasets": {
            "binaryClassificationBaseMainInput": {
                "idField": "id",
                "targetField": "target",
                "targetClass": 1,
                "predictorFields": [
                    {
                        "fieldName": "field1",
                        "dataType": "CATEGORICAL"
                    },
                    {
                        "fieldName": "field2",
                        "dataType": "INT"
                    },
                    {
                        "fieldName": "field3",
                        "dataType": "REAL"
                    },
                    {
                        "fieldName": "field4",
                        "dataType": "NUMERIC"
                    }
                ]
            }
        }
    }
    
    # Create an instance of the Schema model with the schema data
    schema = SchemaModel(**sample_schema_dict)

    # Validate the schema by calling its validate() method
    SchemaModel.validate(sample_schema_dict)

    print("all good")