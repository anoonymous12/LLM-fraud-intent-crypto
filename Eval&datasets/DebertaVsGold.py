import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


# 1. Load gold dataset
gold_path = "mixed_gold_intent_split_1_200_skewed_75_75_50.csv" #change csv with the right gold csv everytime for more tests.

df = pd.read_csv(gold_path)
df = df[["text", "gold_label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)


# Label mapping (must match training)
label2id = {
    "fraud": 0,
    "prevention": 1,
    "out_of_context": 2,
}
id2label = {v: k for k, v in label2id.items()}

y_true = df["gold_label"].map(label2id).values


# 2. Load trained model
model_dir = "./best_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)

model.eval()


# 3. Batched inference
@torch.no_grad()
def predict(texts, batch_size=32, max_length=256):
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())

    return np.array(predictions)


y_pred = predict(df["text"].tolist())


# 4. Evaluation
accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=[id2label[i] for i in range(len(id2label))],
    digits=4,
    zero_division=0,
))
