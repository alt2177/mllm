from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

def main():
    # set our collective token
    access_token = "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" 
    username = "mllm-dev"
    output_repo = "merge_diff_data_IMDB_10"

    # load and tokenize data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=2,
        use_auth_token = access_token,
	id2label=id2label,
	label2id=label2id
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # create training-val split
    small_train_dataset = tokenized_data["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_data["test"].shuffle(seed=42)
 
    output_dir = "YELP_full"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
	    hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    result = trainer.train()
    print(result)
#    trainer.push_to_hub(f"{username}/{output_repo}")
	
    HfFolder.save_token(access_token)

    # Optionally, you can directly use the token with the HfApi instance
    api = HfApi()
    user = api.whoami(token=access_token)
    print("Logged in as:", user['name'])
    
    print('UPLOADING')
    api.upload_folder(
        folder_path=f"./{output_dir}",
        repo_id=f"{username}/{output_repo}",
        repo_type="model"
    )
    print('uploading done!')


    # model eval
    #eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])


    #print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")



if __name__ == "__main__":
    main()
