import os
import sys
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from src.components.data_transformation import DataTransformation
from src.logger.logging import logging
from src.exception.exception import CustomException


# ---------------------- CONFIG ----------------------
@dataclass
class ModelTrainerConfig:
    pretrained_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    num_labels: int = 2
    output_dir: str = "models/"
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    max_len: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- LABEL FUNCTION ----------------------
def get_label(compound):
    if compound >= 0.05:
        return 1
    elif compound <= -0.05:
        return 0
    else:
        return None


# ---------------------- DATASET CLASS ----------------------
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["Text"].astype(str).values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------- TRAINER CLASS ----------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_model(self, dataset_path):
        try:
            logging.info("Loading dataset...")
            df = pd.read_csv(dataset_path)

            # Create binary labels
            df["label"] = df["compound"].apply(get_label)
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)

            logging.info(f"Dataset size after filtering: {len(df)}")

            # Train test split
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

            logging.info("Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name)

            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.pretrained_model_name,
                num_labels=self.config.num_labels,
            )

            model.to(self.config.device)

            # Create datasets
            train_dataset = ReviewDataset(train_df, tokenizer, self.config.max_len)
            val_dataset = ReviewDataset(val_df, tokenizer, self.config.max_len)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

            optimizer = AdamW(model.parameters(), lr=self.config.lr)

            logging.info("Starting training...")

            best_acc = 0

            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].to(self.config.device)
                    attention_mask = batch["attention_mask"].to(self.config.device)
                    labels = batch["labels"].to(self.config.device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                logging.info(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

                # ---------------- VALIDATION ----------------
                model.eval()
                preds, true_labels = [], []

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.config.device)
                        attention_mask = batch["attention_mask"].to(self.config.device)
                        labels = batch["labels"].to(self.config.device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=1)

                        preds.extend(predictions.cpu().numpy())
                        true_labels.extend(labels.cpu().numpy())

                acc = accuracy_score(true_labels, preds)
                logging.info(f"Epoch {epoch+1} Validation Accuracy: {acc:.4f}")

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    model.save_pretrained(self.config.output_dir)
                    tokenizer.save_pretrained(self.config.output_dir)
                    logging.info("✅ Best model saved!")

            logging.info(f"Training complete! Best Accuracy: {best_acc:.4f}")

        except Exception as e:
            raise CustomException(e, sys)
from src.components.data_ingestion import DataIngestion

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    trns = DataTransformation()
    train_arr, test_arr = trns.initiate_data_transformation()
    model_trainer = ModelTrainer()
    model_trainer.train_model(dataset_path="data/reviews.csv")