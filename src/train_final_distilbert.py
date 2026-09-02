import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# DARPAN - FINAL DISTILBERT TRAINING
# ============================================================

print("=" * 75)
print("DARPAN - FINAL DISTILBERT TRAINING")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "darpan_final_splits"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "darpan_final_distilbert"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.csv"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading final datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train samples      :", len(train_df))
print("Validation samples :", len(val_df))
print("Test samples       :", len(test_df))


# ============================================================
# LABEL MAPPING
# ============================================================

labels = sorted(
    train_df["label"].unique()
)

label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

print("\nLabels:")

for label, idx in label2id.items():
    print(
        f"{idx}: {label}"
    )


# ============================================================
# DATASET
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


class DARPANDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=256
    ):

        self.texts = (
            dataframe["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .map(label2id)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }


train_dataset = DARPANDataset(
    train_df,
    tokenizer
)

val_dataset = DARPANDataset(
    val_df,
    tokenizer
)

test_dataset = DARPANDataset(
    test_df,
    tokenizer
)


# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if device.type == "cpu":
    print("CPU training enabled.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

model.to(device)


# ============================================================
# LOSS
# ============================================================

loss_function = torch.nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

LEARNING_RATE = 2e-5

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 5

best_val_f1 = -1.0


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels_batch = (
                batch["labels"]
                .to(device)
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = loss_function(
                logits,
                labels_batch
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    avg_loss = (
        total_loss / len(loader)
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 75)
print("STARTING FINAL TRAINING")
print("=" * 75)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 50)

    model.train()

    total_train_loss = 0.0

    for batch_idx, batch in enumerate(
        train_loader
    ):

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        labels_batch = (
            batch["labels"]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        loss = loss_function(
            logits,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        if (
            (batch_idx + 1) % 20
            == 0
        ):

            print(
                f"Batch {batch_idx + 1}/"
                f"{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    avg_train_loss = (
        total_train_loss
        / len(train_loader)
    )

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    print(
        "\nTraining Loss   :",
        round(avg_train_loss, 4)
    )

    print(
        "Validation Loss :",
        round(val_loss, 4)
    )

    print(
        "Validation Acc  :",
        round(val_accuracy, 4)
    )

    print(
        "Validation Prec :",
        round(val_precision, 4)
    )

    print(
        "Validation Rec  :",
        round(val_recall, 4)
    )

    print(
        "Validation F1   :",
        round(val_f1, 4)
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        print(
            "\n*** New best model! "
            "Saving... ***"
        )

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        with open(
            os.path.join(
                MODEL_DIR,
                "label_mapping.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "label2id": label2id,
                    "id2label": id2label
                },
                f,
                indent=4
            )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_DIR)
)

model.to(device)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)


print(
    "\nTest Loss      :",
    round(test_loss, 4)
)

print(
    "Test Accuracy  :",
    round(test_accuracy, 4)
)

print(
    "Test Precision :",
    round(test_precision, 4)
)

print(
    "Test Recall    :",
    round(test_recall, 4)
)

print(
    "Test F1        :",
    round(test_f1, 4)
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    test_labels,
    test_predictions,
    target_names=labels,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model":
        MODEL_NAME,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "train_samples":
        len(train_df),

    "validation_samples":
        len(val_df),

    "test_samples":
        len(test_df),

    "best_validation_f1":
        float(best_val_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_precision":
        float(test_precision),

    "test_recall":
        float(test_recall),

    "test_f1":
        float(test_f1)
}

with open(
    os.path.join(
        MODEL_DIR,
        "training_results.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

cm_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 75)
print("FINAL TRAINING COMPLETED")
print("=" * 75)

print(
    "\nBest model:",
    MODEL_DIR
)

print(
    "\nTraining results:",
    os.path.join(
        MODEL_DIR,
        "training_results.json"
    )
)

print(
    "\nConfusion matrix:",
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.csv"
    )
)