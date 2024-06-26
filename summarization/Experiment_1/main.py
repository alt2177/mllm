from huggingface_hub import HfApi, HfFolder, create_repo
<<<<<<< HEAD
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
=======
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np
import torch
>>>>>>> main

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

<<<<<<< HEAD
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

def main():
    # set our collective token
    access_token = "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" 
    username = "mllm-dev"
    output_repo = "gpt2_finetune_2"
=======
def main():
    # set our collective token and HF info
    access_token = "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" 
    username = "mllm-dev"
    # repo that will be made on huggingface
    output_repo = "yelp_finetuned_6gpu_full"
>>>>>>> main

    # load and tokenize data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

<<<<<<< HEAD
    tokenized_imdb = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

=======
    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

>>>>>>> main
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

<<<<<<< HEAD
    # create training-val split
    #small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(10))
    #small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(10))
 
    output_dir = "imdb_finetune_gpt2_test_epoch_5"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
=======
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
>>>>>>> main
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
<<<<<<< HEAD
        push_to_hub=True,
	    hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token
=======
	    hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token,
	    push_to_hub=True
>>>>>>> main
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
<<<<<<< HEAD
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


=======
        compute_metrics=compute_metrics,
    )
    print('************TRAINING STARTED*************')
    trainer.train()
    print('*************TRAINING COMPLETED**************')
   # trainer.evaluate()
>>>>>>> main

if __name__ == "__main__":
    main()
