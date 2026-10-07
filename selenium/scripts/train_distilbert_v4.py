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


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "../../"

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "train_multilabel_v4.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "validation_multilabel_v4.csv"
)

MODEL_OUTPUT = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_dark_pattern_v4_multilabel"
)

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256


LABELS = [
    "Sneaking",
    "hidden fees",
    "interface interference",
    "misdirection",
    "obstruction",
    "potential false urgency",
]

NUM_LABELS = len(LABELS)


LABEL2ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DARPAN - DISTILBERT V4 MULTI-LABEL TRAINING")
print("=" * 70)

print("\nLabels:")

for i, label in enumerate(LABELS):
    print(f"{i} -> {label}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


print(
    f"Training samples   : {len(train_df)}"
)

print(
    f"Validation samples : {len(val_df)}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "text"
] + LABELS

for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in val_df.columns:

        raise ValueError(
            f"Missing validation column: {column}"
        )


# ============================================================
# CLEAN TEXT
# ============================================================

train_df["text"] = (
    train_df["text"]
    .fillna("")
    .astype(str)
)

val_df["text"] = (
    val_df["text"]
    .fillna("")
    .astype(str)
)


# ============================================================
# CREATE MULTI-LABEL VECTORS
# ============================================================

train_df["labels"] = train_df[
    LABELS
].astype(float).values.tolist()

val_df["labels"] = val_df[
    LABELS
].astype(float).values.tolist()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nTRAINING:")

for label in LABELS:

    print(
        f"{label:<30}: "
        f"{train_df[label].sum()}"
    )


print("\nVALIDATION:")

for label in LABELS:

    print(
        f"{label:<30}: "
        f"{val_df[label].sum()}"
    )


# ============================================================
# CALCULATE POSITIVE WEIGHTS
#
# BCEWithLogitsLoss:
#
# weight = negative_samples / positive_samples
#
# This is especially important because potential false
# urgency has relatively few positive samples.
# ============================================================

print("\n")
print("=" * 70)
print("CALCULATING POSITIVE WEIGHTS")
print("=" * 70)

positive_weights = []

for label in LABELS:

    positive_count = train_df[label].sum()

    negative_count = (
        len(train_df) - positive_count
    )

    if positive_count == 0:

        weight = 1.0

    else:

        weight = (
            negative_count /
            positive_count
        )

    positive_weights.append(weight)

    print(
        f"{label:<30}: {weight:.4f}"
    )


pos_weight = torch.tensor(
    positive_weights,
    dtype=torch.float
)


# ============================================================
# HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[
        ["text", "labels"]
    ],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[
        ["text", "labels"]
    ],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

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


print("\nTokenizing...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# FORMAT
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

print("\nLoading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    problem_type="multi_label_classification",
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    # --------------------------------------------------------
    # MULTI-LABEL:
    #
    # sigmoid converts every output independently to 0..1
    # --------------------------------------------------------

    probabilities = 1 / (
        1 + np.exp(-logits)
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    labels = labels.astype(int)

    # --------------------------------------------------------
    # MICRO METRICS
    # --------------------------------------------------------

    precision_micro, recall_micro, f1_micro, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # MACRO METRICS
    # --------------------------------------------------------

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # EXACT MATCH
    #
    # Entire six-label vector must be correct.
    # --------------------------------------------------------

    exact_match = np.mean(
        np.all(
            predictions == labels,
            axis=1
        )
    )

    return {
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "exact_match": exact_match,
    }


# ============================================================
# MULTI-LABEL TRAINER
# ============================================================

class MultiLabelTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs
    ):

        labels = inputs.pop("labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        logits = outputs.logits

        weights = pos_weight.to(
            logits.device
        )

        # ----------------------------------------------------
        # THE IMPORTANT CHANGE:
        #
        # BCEWithLogitsLoss
        #
        # NOT CrossEntropyLoss
        # ----------------------------------------------------

        loss_function = (
            torch.nn.BCEWithLogitsLoss(
                pos_weight=weights
            )
        )

        loss = loss_function(
            logits,
            labels.float()
        )

        if return_outputs:

            return loss, outputs

        return loss


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT,

    num_train_epochs=5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_steps=10,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=10,

    load_best_model_at_end=True,

    metric_for_best_model="micro_f1",

    greater_is_better=True,

    fp16=False,

    dataloader_pin_memory=False,

    seed=42,

    report_to="none",

    save_total_limit=2,
)


# ============================================================
# TRAINER
# ============================================================

trainer = MultiLabelTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING V4 MULTI-LABEL TRAINING")
print("=" * 70)

train_result = trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("V4 MULTI-LABEL VALIDATION")
print("=" * 70)

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
# SAVE
# ============================================================

print("\n")
print("=" * 70)
print("SAVING V4 MULTI-LABEL MODEL")
print("=" * 70)

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
print("=" * 70)
print("DISTILBERT V4 MULTI-LABEL TRAINING COMPLETE")
print("=" * 70)