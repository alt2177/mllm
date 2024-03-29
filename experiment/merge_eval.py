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

#small_eval_dataset = tokenized_yelp["test"]

eval_dataset = tokenized_yelp["test"].shuffle(seed=42)

# Define the size of the validation set (e.g., 20% of the total data)
validation_size = int(0.2 * len(eval_dataset))
indices = list(range(len(eval_dataset)))
test_indices = indices[validation_size:]
validation_indices = indices[:validation_size]

# Split the shuffled training dataset into training and validation se
test_dataset = eval_dataset.select(test_indices)
validation_dataset = eval_dataset.select(validation_indices)


trainer = Trainer(
    model=model,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


validation_result = trainer.evaluate(eval_dataset=validation_dataset)
print("Merge validation:", validation_result)

test_result = trainer.evaluate(eval_dataset=test_dataset)

f = open("accuracy_merge.txt", "a")
f.write(f"DARE Ties merge validation results : {validation_result}\n")
f.write(f"DARE Ties test results : {test_result}\n\n")
f.close()




