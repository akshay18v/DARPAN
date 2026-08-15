import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "../../data/metadata/train.csv"
VAL_PATH = "../../data/metadata/validation.csv"
TEST_PATH = "../../data/metadata/test.csv"

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "../../models/distilbert_dark_pattern"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DARPAN - DISTILBERT TRAINING")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDataset sizes:")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))


# ============================================================
# LABEL MAPPING
# ============================================================

labels = sorted(train_df["category"].unique())

label2id = {
    label: idx
    for idx, label in enumerate(labels)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

print("\nLabels:")

for idx, label in id2label.items():
    print(idx, "->", label)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe(df):
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str)

    df["label"] = df["category"].map(label2id)

    return df[["text", "label"]]


train_df = prepare_dataframe(train_df)
val_df = prepare_dataframe(val_df)
test_df = prepare_dataframe(test_df)


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels_true = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels_true,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=5,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_steps=10,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\nValidation results:")

validation_results = trainer.evaluate(
    val_dataset
)

print(validation_results)


# ============================================================
# TEST
# ============================================================

print("\nTest results:")

test_results = trainer.evaluate(
    test_dataset
)

print(test_results)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 60)
print("DISTILBERT TRAINING COMPLETE")
print("=" * 60)