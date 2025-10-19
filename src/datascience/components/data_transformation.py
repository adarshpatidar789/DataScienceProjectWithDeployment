import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import DataTransformationConfig
import pandas as pd

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data = None
    
    def categorical_to_numerical(self):
        data = pd.read_csv(self.config.data_path)
        mapping_home_ownership = {'RENT': 1, 'MORTGAGE': 2, 'OWN': 3, 'OTHER': 0}
        mapping_loan_grade = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F":6, "G":6}
        mapping_cb_person_default_on_file = {"N": 1, "Y": 0}

        data['person_home_ownership'] = data['person_home_ownership'].map(mapping_home_ownership).fillna(0)
        data['loan_grade'] = data['loan_grade'].map(mapping_loan_grade).fillna(0)
        data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map(mapping_cb_person_default_on_file).fillna(0)

        # # one-hot incoding for loan_intent as can't specify order/let the model decide
        # data = pd.get_dummies(data, columns=['loan_intent'], drop_first=True)

        data = data.drop('loan_intent', axis=1)

        self.data = data  # make it available for other methods
        logger.info("Categorical variables are sucessfully converted to numerical.")

    def train_test_splitting(self):
        if self.data is None:
            raise ValueError("Data not found! Run categorical_to_numerical() first.")
        
        train, test = train_test_split(self.data, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)

    
        

