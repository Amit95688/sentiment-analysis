
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger.logging import logging
from src.exception.exception import CustomException
# Optional: from src.config.mlflow_config import MLflowConfig
import pandas as pd

def run_training_pipeline(use_hyperparameter_tuning=False, use_pytorch=False):
	try:
		# Setup MLflow (optional)
		# MLflowConfig.setup_mlflow()

		logging.info("="*60)
		logging.info("TRAINING PIPELINE STARTED")
		logging.info("="*60)

		# Step 1: Data Ingestion
		logging.info("Step 1: Data Ingestion")
		data_ingestion = DataIngestion()
		train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
		logging.info(f"✓ Data ingestion completed | Train: {train_data_path} | Test: {test_data_path}")
		mlflow.log_param("train_data_path", train_data_path)
		mlflow.log_param("test_data_path", test_data_path)

		# Step 2: Data Transformation
		logging.info("Step 2: Data Transformation")
		data_transformation = DataTransformation()
		train_pkl, test_pkl = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
		logging.info(f"✓ Data transformation completed | Train: {train_pkl} | Test: {test_pkl}")
		mlflow.log_param("train_pkl", train_pkl)
		mlflow.log_param("test_pkl", test_pkl)

		# Step 3: Model Training
		logging.info("Step 3: Model Training")
		trainer = ModelTrainer()
		trainer.train_model(train_pkl, test_pkl)
		logging.info("✓ Model training completed")
		# Optionally log model path, metrics, etc.

		logging.info("="*60)
		logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
		logging.info("="*60)
		return {
			'train_data': train_data_path,
			'test_data': test_data_path,
			'train_pkl': train_pkl,
			'test_pkl': test_pkl,
			'status': 'success'
		}
	except Exception as e:
		logging.error(f"❌ Training pipeline failed: {str(e)}")
		raise CustomException(e, sys)

if __name__ == "__main__":
	result = run_training_pipeline(
		use_hyperparameter_tuning=False,
		use_pytorch=False
	)
	print("\n✓ Training pipeline completed!")
	print(f"Train data: {result['train_data']}")
	print(f"Test data: {result['test_data']}")
	print(f"Train pkl: {result['train_pkl']}")
	print(f"Test pkl: {result['test_pkl']}")
