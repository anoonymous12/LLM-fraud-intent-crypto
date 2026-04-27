import pandas as pd
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. Load dataset
df = pd.read_csv("cleaned_dataset.csv")  # expected columns: text, label / these are the silver_labeled data 130.000 comments from stackexchange, avalaible in this repository also 
df = df[["text", "label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# In the CSV, labels are stored as:
#  - "fraud"          → corresponds to the "to attack" intent
#  - "prevention"     → corresponds to the "to prevent" intent
#  - "out_of_context" → corresponds to the out-of-context class
label2id = {
    "fraud": 0,
    "prevention": 1,
    "out_of_context": 2,
}
id2label = {v: k for k, v in label2id.items()}

df = df[df["label"].isin(label2id)]
df["label"] = df["label"].map(label2id)


# 2. Train/validation split
df_train, df_val = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"],
)

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
})


# 3. Model and tokenizer
model_name = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


# 4. Tokenization
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)

encoded_dataset = encoded_dataset.remove_columns(
    [c for c in encoded_dataset["train"].column_names
     if c not in ["input_ids", "attention_mask", "label", "token_type_ids"]]
)

encoded_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 5. Class weights
classes = np.array(sorted(df_train["label"].unique()))
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df_train["label"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float)


# 6. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


# 7. Custom trainer (weighted loss)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# 8. Training configuration
training_args = TrainingArguments(
    output_dir="./runs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none",
    seed=SEED,
)


# 9. Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# 10. Training
trainer.train()


# 11. Evaluation
metrics = trainer.evaluate()

pred_output = trainer.predict(encoded_dataset["validation"])
preds = np.argmax(pred_output.predictions, axis=-1)
labels = pred_output.label_ids

print("Evaluation metrics:", metrics)
print("\nClassification report:")
print(classification_report(
    labels,
    preds,
    target_names=[id2label[i] for i in range(len(id2label))],
    digits=4
))

print("\nConfusion matrix:")
print(confusion_matrix(labels, preds))


# 12. Save model
save_dir = "./best_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
