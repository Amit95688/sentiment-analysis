import os 
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception.exception import CustomException
from src.logger.logging import logging

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
            chunk_size = 10000  # Adjust as needed for memory
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)

            def tokenize_and_save(input_path, output_path):
                chunk_iter = pd.read_csv(input_path, chunksize=chunk_size)
                first_chunk = True
                for chunk in chunk_iter:
                    # Tokenize the 'Text' column in this chunk
                    chunk['Text'] = self.tokenizer(chunk['Text'].tolist(), padding=True, truncation=True, return_tensors="pt")['input_ids'].tolist()
                    # Save chunk to pickle (append mode)
                    if first_chunk:
                        chunk.to_pickle(output_path)
                        first_chunk = False
                    else:
                        # Append to pickle by reading, concatenating, and saving (since pickle doesn't support append)
                        prev = pd.read_pickle(output_path)
                        pd.concat([prev, chunk], ignore_index=True).to_pickle(output_path)

            logging.info("Tokenizing and saving train data in chunks")
            tokenize_and_save(train_path, self.data_transformation_config.transformed_train_path)
            logging.info("Tokenizing and saving test data in chunks")
            tokenize_and_save(test_path, self.data_transformation_config.transformed_test_path)
            logging.info("Transformed data saved successfully (chunked)")
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