import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "../../"

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "metadata", "train_v3.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR, "data", "metadata", "validation_v3.csv"
)

MODEL_OUTPUT = os.path.join(
    BASE_DIR, "models", "distilbert_dark_pattern_v3"
)

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256

LABELS = [
    "Sneaking",
    "hidden fees",
    "interface interference",
    "misdirection",
    "obstruction",
<<<<<<< HEAD
    "potential false urgency"
=======
>>>>>>> a207492 (training and urgency detector)
]

LABEL2ID = {
    label: i for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label for i, label in enumerate(LABELS)
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("DARPAN - DISTILBERT V3 TRAINING")
print("=" * 60)

print("\nLabels:")

for i, label in enumerate(LABELS):
    print(f"{i} -> {label}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


# Remove rows with missing text/category
train_df = train_df.dropna(subset=["text", "category"]).copy()
val_df = val_df.dropna(subset=["text", "category"]).copy()


# Normalize text
train_df["text"] = train_df["text"].astype(str)
val_df["text"] = val_df["text"].astype(str)


# Normalize category spacing/case only where necessary
train_df["category"] = train_df["category"].str.strip()
val_df["category"] = val_df["category"].str.strip()


# Make sure categories are valid
invalid_train = set(train_df["category"]) - set(LABELS)
invalid_val = set(val_df["category"]) - set(LABELS)

if invalid_train:
    raise ValueError(
        f"Invalid training labels found: {invalid_train}"
    )

if invalid_val:
    raise ValueError(
        f"Invalid validation labels found: {invalid_val}"
    )


print("\nDataset sizes:")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(
    train_df["category"]
    .value_counts()
)


# ============================================================
# CONVERT LABELS TO INTEGER IDs
# ============================================================

train_df["labels"] = train_df["category"].map(LABEL2ID)
val_df["labels"] = val_df["category"].map(LABEL2ID)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes = np.arange(len(LABELS))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["labels"].values,
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)


print("\nClass weights:")

for i, label in enumerate(LABELS):
    print(
        f"{label:<25} -> {class_weights[i].item():.4f}"
    )


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "labels"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "labels"]],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
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


# ============================================================
# FORMAT DATASETS
# ============================================================

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels",
    ]
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


# ============================================================
# METRICS
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
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs
    ):

        labels = inputs.get("labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        logits = outputs.logits

        weights = class_weights.to(
            logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits,
            labels
        )

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT,

    # Training
    num_train_epochs=5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_steps=10,

    # Evaluation
    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",
    logging_steps=10,

    # Select best checkpoint
    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    # CPU
    fp16=False,

    dataloader_pin_memory=False,

    # Reproducibility
    seed=42,

    # Reporting
    report_to="none",

    # Keep only best few checkpoints
    save_total_limit=2,
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING V3 TRAINING")
print("=" * 60)

train_result = trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("V3 VALIDATION RESULTS")
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
# SAVE BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("SAVING BEST V3 MODEL")
print("=" * 60)

os.makedirs(
    MODEL_OUTPUT,
    exist_ok=True
)

trainer.save_model(
    MODEL_OUTPUT
)

tokenizer.save_pretrained(
    MODEL_OUTPUT
)


print("\nModel saved to:")
print(MODEL_OUTPUT)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("DISTILBERT V3 TRAINING COMPLETE")
print("=" * 60)