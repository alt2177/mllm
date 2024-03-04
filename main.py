<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np
import torch

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # set our collective token and HF info
    access_token = "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" 
    username = "mllm-dev"
    # repo that will be made on huggingface
    output_repo = "yelp_finetuned_6gpu_full"
=======

=======
import os
>>>>>>> 4565eef23 (began trying to use mergekit in our data pipeline)
import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
=======
#from huggingface_hub import notebook_login
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
>>>>>>> 82f519b2f (created sbatch file)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    fs = pyxet.XetFS()
    print("hello sheep")
>>>>>>> 98e6ae202 (created stuff)

    # load and tokenize data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token
=======
    df_training, df_testing = load_drug_data()
    print(df_training.head())
>>>>>>> 2503f0e0a (got imports to work)
=======
    os.chdir("mergekit")
    df_training, df_testing = load_drug_data()
    print(df_training.head())
    model = MLLM()
>>>>>>> 4565eef23 (began trying to use mergekit in our data pipeline)
=======
    # set tam's token
    access_token = "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" 

    # load and tokenize data
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=5,
        token = access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # create training-val split
    small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(100))
 
    # training loop
    training_args = TrainingArguments(
        output_dir="yelp_finetune_epoch_1_gpt2_1",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
<<<<<<< HEAD
        hub_token=access_tokens,
=======
        hub_token = access_token
>>>>>>> e8a460892 (model trained)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
<<<<<<< HEAD
        hub_token=access_tokens,
=======
#        hub_token = access_token
>>>>>>> e8a460892 (model trained)
    )

    result = trainer.train()
    print(result)

    # model eval
    #eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])


    #print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")

>>>>>>> 82f519b2f (created sbatch file)

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

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

    small_train_dataset = tokenized_yelp["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42)

    # create the repo before we try to push the model to huggingface
    HfFolder.save_token(access_token)
    api = HfApi()
    user = api.whoami(token=access_token)
    try:
    	create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "yelp_finetune_gpt2_test"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
        #gradient_accumulation_steps=4,   #want to explore what this is
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=2,
	    fp16=True,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
	    hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token,
	    push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print('************TRAINING STARTED*************')
    trainer.train()
    print('*************TRAINING COMPLETED**************')
   # trainer.evaluate()

if __name__ == "__main__":
    main()
