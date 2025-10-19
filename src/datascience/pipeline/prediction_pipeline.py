import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))

    def encode_categorical_features(self, data: pd.DataFrame):
        """Apply same categorical mappings as in training."""
        mapping_home_ownership = {'RENT': 1, 'MORTGAGE': 2, 'OWN': 3, 'OTHER': 0}
        mapping_loan_grade = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 6}
        mapping_cb_person_default_on_file = {"N": 1, "Y": 0}

        data['person_home_ownership'] = data['person_home_ownership'].map(mapping_home_ownership).fillna(0)
        data['loan_grade'] = data['loan_grade'].map(mapping_loan_grade).fillna(0)
        data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map(mapping_cb_person_default_on_file).fillna(0)

        # Drop loan_intent if you did so during training
        if 'loan_intent' in data.columns:
            data = data.drop('loan_intent', axis=1)

        return data

    def predict(self, data):
        """Predict on raw input data (dict or DataFrame)."""
        # Ensure DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])

        # Apply same categorical mappings
        data = self.encode_categorical_features(data)

        # Make prediction
        prediction = self.model.predict(data)
        return prediction
