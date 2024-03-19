from datasets import load_dataset
dataset = load_dataset("yelp_review_full")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("mllm-dev/gpt2_m_experiment")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

tokenized_yelp = dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "mllm-dev/gpt2_m_experiment", num_labels=5
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

small_eval_dataset = tokenized_yelp["test"]

trainer = Trainer(
    model=model,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

eval_result = trainer.evaluate(eval_dataset=small_eval_dataset)

print(eval_result)
