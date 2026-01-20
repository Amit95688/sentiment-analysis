import os 
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception.exception import CustomException
from src.logger.logging import logging

from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType
import torch
from transformers import AutoTokenizer

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join('artifacts', 'preprocessed_train.pkl')
    transformed_test_path: str = os.path.join('artifacts', 'preprocessed_test.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts (Spark)")
        try:
            spark = SparkSession.builder.appName("DataTransformation").getOrCreate()
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)

            def process_and_save(input_path, output_path):
                df = spark.read.csv(input_path, header=True)
                # Register UDF for tokenization
                tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
                def tokenize_udf(text):
                    if text is None:
                        return []
                    return tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")['input_ids'][0].tolist()
                spark_tokenize_udf = udf(tokenize_udf, ArrayType(IntegerType()))
                df = df.withColumn("Text_tokenized", spark_tokenize_udf(df["Text"]))
                df.write.mode("overwrite").parquet(output_path)

            logging.info("Tokenizing and saving train data with Spark")
            process_and_save(train_path, self.data_transformation_config.transformed_train_path)
            logging.info("Tokenizing and saving test data with Spark")
            process_and_save(test_path, self.data_transformation_config.transformed_test_path)
            logging.info("Transformed data saved successfully (Spark)")
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