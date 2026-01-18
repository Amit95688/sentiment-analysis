import os
import sys
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from src.logger.logging import logging
from src.exception.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion


# ---------------------- CONFIG ----------------------
@dataclass
class ModelTrainerConfig:
    pretrained_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    num_labels: int = 2
    output_dir: str = "models/"
    batch_size: int = 8  # Reduced batch size to lower memory usage
    lr: float = 2e-5
    epochs: int = 3
    max_len: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- DATASET ----------------------
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------- TRAINER ----------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_model(self, train_pkl_path, val_pkl_path):
        try:
            logging.info("Loading transformed datasets...")
            train_df = pd.read_pickle(train_pkl_path)
            val_df = pd.read_pickle(val_pkl_path)

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.pretrained_model_name,
                num_labels=self.config.num_labels,
            ).to(self.config.device)

            train_dataset = ReviewDataset(
                train_df, tokenizer, self.config.max_len
            )
            val_dataset = ReviewDataset(
                val_df, tokenizer, self.config.max_len
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,  # Increased for faster data loading
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,  # Increased for faster data loading
                pin_memory=True
            )

            optimizer = AdamW(
                model.parameters(),
                lr=self.config.lr
            )

            best_acc = 0.0
            logging.info("🚀 Training started")

            scaler = torch.cuda.amp.GradScaler() if self.config.device == "cuda" else None
            for epoch in range(self.config.epochs):
                # -------- TRAINING --------
                model.train()
                total_loss = 0.0

                for batch in train_loader:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].to(self.config.device)
                    attention_mask = batch["attention_mask"].to(self.config.device)
                    labels = batch["labels"].to(self.config.device)

                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                        loss = outputs.loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item() if not scaler else loss.item()

                avg_loss = total_loss / len(train_loader)
                logging.info(
                    f"Epoch {epoch+1}/{self.config.epochs} "
                    f"Training Loss: {avg_loss:.4f}"
                )

                # -------- VALIDATION --------
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

                        predictions = torch.argmax(
                            outputs.logits, dim=1
                        )

                        preds.extend(predictions.cpu().numpy())
                        true_labels.extend(labels.cpu().numpy())

                acc = accuracy_score(true_labels, preds)
                logging.info(
                    f"Epoch {epoch+1} Validation Accuracy: {acc:.4f}"
                )

                # -------- SAVE BEST MODEL --------
                if acc > best_acc:
                    best_acc = acc
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    model.save_pretrained(self.config.output_dir)
                    tokenizer.save_pretrained(self.config.output_dir)
                    logging.info("✅ Best model saved")

                # -------- MEMORY CLEANUP --------
                import gc
                del batch, input_ids, attention_mask, labels, outputs
                gc.collect()
                if self.config.device == "cuda":
                    torch.cuda.empty_cache()

            logging.info(
                f"🎉 Training complete | Best Accuracy: {best_acc:.4f}"
            )

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_pkl, test_pkl = transformation.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    trainer.train_model(train_pkl, test_pkl)
