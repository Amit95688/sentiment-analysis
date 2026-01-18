import os 
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception.exception import CustomException
from src.logger.logging import logging
 # Do not import DataIngestion at the top to avoid circular import
from dataclasses import dataclass
import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join('artifacts', 'preprocessed_train.pkl')
    transformed_test_path: str = os.path.join('artifacts', 'preprocessed_test.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts")
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Training and Testing data read successfully")

            train_df['Text'] = self.tokenizer(train_df['Text'].tolist(), padding=True, truncation=True, return_tensors="pt")['input_ids'].tolist()
            test_df['Text'] = self.tokenizer(test_df['Text'].tolist(), padding=True, truncation=True, return_tensors="pt")['input_ids'].tolist()
            logging.info("Text data tokenized successfully")

            # Save the transformed data
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)
            train_df.to_pickle(self.data_transformation_config.transformed_train_path)
            test_df.to_pickle(self.data_transformation_config.transformed_test_path)
            logging.info("Transformed data saved successfully")
            return (
                self.data_transformation_config.transformed_train_path,
                self.data_transformation_config.transformed_test_path
            )
        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion  # Import here to avoid circular import
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    transformed_train_path, transformed_test_path = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )