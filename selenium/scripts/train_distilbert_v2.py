import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# DARPAN - DISTILBERT V2 TRAINING
# ============================================================

print("=" * 60)
print("DARPAN - DISTILBERT V2 TRAINING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

TRAIN_PATH = "../../data/metadata/train_clean.csv"
VAL_PATH = "../../data/metadata/validation_clean.csv"

MODEL_OUTPUT = "../../models/distilbert_dark_pattern_v2"

MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# 2. LABEL MAPPING
# ============================================================

LABELS = [
    "Sneaking",
    "hidden fees",
    "interface interference",
    "misdirection",
    "obstruct_sneaking",
    "obstruction",
]

label2id = {
    label: i
    for i, label in enumerate(LABELS)
}

id2label = {
    i: label
    for i, label in enumerate(LABELS)
}

print("\nLabels:")

for i, label in enumerate(LABELS):
    print(f"{i} -> {label}")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


# Remove accidental missing values
train_df = train_df.dropna(subset=["text", "category"]).copy()
val_df = val_df.dropna(subset=["text", "category"]).copy()


# Make sure labels are valid
train_df = train_df[
    train_df["category"].isin(LABELS)
].copy()

val_df = val_df[
    val_df["category"].isin(LABELS)
].copy()


# Convert category -> integer label
train_df["label"] = train_df["category"].map(label2id)
val_df["label"] = val_df["category"].map(label2id)


print("\nDataset sizes:")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")


# ============================================================
# 4. SHOW CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(
    train_df["category"].value_counts()
)


# ============================================================
# 5. CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = np.bincount(
    train_df["label"],
    minlength=len(LABELS)
)

total_samples = len(train_df)
num_classes = len(LABELS)

class_weights = total_samples / (
    num_classes * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("\nClass weights:")

for i, label in enumerate(LABELS):
    print(
        f"{label:25s} -> "
        f"{class_weights[i].item():.4f}"
    )


# ============================================================
# 6. CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# 7. TOKENIZER
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)


# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(
    ["text"]
)

val_dataset = val_dataset.remove_columns(
    ["text"]
)


# ============================================================
# 8. LOAD MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)


# ============================================================
# 9. METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# 10. CUSTOM TRAINER
#     CLASS-WEIGHTED LOSS
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.get("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.get("logits")

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT,

    # Training
    num_train_epochs=5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_ratio=0.1,

    # Evaluation
    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",
    logging_steps=10,

    # Select best model
    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    # Prevent excessive checkpoints
    save_total_limit=2,

    # CPU-friendly
    fp16=False,

    report_to="none",

    # Reproducibility
    seed=42,
)


# ============================================================
# 12. TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ],
)


# ============================================================
# 13. START TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING V2 TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# 14. VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("V2 VALIDATION RESULTS")
print("=" * 60)

results = trainer.evaluate()

print("\nValidation results:")

for key, value in results.items():

    if isinstance(value, float):
        print(
            f"{key}: {value:.4f}"
        )

    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("SAVING BEST V2 MODEL")
print("=" * 60)

trainer.save_model(
    MODEL_OUTPUT
)

tokenizer.save_pretrained(
    MODEL_OUTPUT
)

print(
    f"\nModel saved to:\n{MODEL_OUTPUT}"
)


print("\n")
print("=" * 60)
print("DISTILBERT V2 TRAINING COMPLETE")
print("=" * 60)