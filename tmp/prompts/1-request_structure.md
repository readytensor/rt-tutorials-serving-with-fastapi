Finalized structure:

For classification:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-04-29T18:30:00Z",
  "requestId": "a1b2c3d4e5",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [1.0, 0.0]
    }
  ]
}
```

For regression:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-04-29T18:30:00Z",
  "requestId": "a1b2c3d4e5",
  "targetDescription": "some description",
  "predictions": [
    {
      "sampleId": "4656",
      "prediction": 5.78058
    }
  ]
}
```
