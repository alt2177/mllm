from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np
import torch
from torch.nn.functional import softmax
import os


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # set our collective token and HF info
    access_token = "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" 
    username = "mllm-dev"

    # load and tokenize data
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 
    
    for m in range(5):
    # create model
        model = AutoModelForSequenceClassification.from_pretrained(
            "openai-community/gpt2",
            num_labels=5,
            use_auth_token = access_token
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        # repo that will be made on huggingface
        output_repo = f"gpt2_f_experiment_{m}"
        # Select 5% of the train data for finetuning a specific model
        num_of_samples = int(len(tokenized_yelp["train"]) * 0.05)
        # print(num_of_samples) # 32500

        small_train_dataset = tokenized_yelp["train"].shuffle(seed=m).select(range(num_of_samples))
        small_eval_dataset = tokenized_yelp["test"]

        # create the repo before we try to push the model to huggingface
        HfFolder.save_token(access_token)
        api = HfApi()
        user = api.whoami(token=access_token)
        try:
    	    create_repo(f"{username}/{output_repo}", repo_type="model")
        except:
            print('error creating repo for model. it probably already exists')

        output_dir = "sean_test_out"
        # training loop
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
            per_device_train_batch_size=24,
            per_device_eval_batch_size=24,
            num_train_epochs=1,
            fp16=True,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            hub_model_id=f"{username}/{output_repo}",
            hub_token = access_token,
            push_to_hub=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print('************TRAINING STARTED*************')
        trainer.train()
        print('*************TRAINING COMPLETED**************')
        
        # Save the output probabilities for merging outputs
        result = trainer.train()
        print(result)
        predictions = trainer.predict(small_eval_dataset)
        logits = predictions.predictions
        probs = softmax(torch.tensor(logits).float(),dim=1).numpy()
        torch.save(probs,f"{output_repo}.pt")
        eval_result = trainer.evaluate(eval_dataset=small_eval_dataset)

        # Save the accuracy of each model for later comparison
        f = open("accuracy.txt", "a")
        f.write(f"Fine tune model {m} accuracy : {eval_result['eval_accuracy']}\n")
        f.close()

        # When push to hub is false, the model is saved under a folder that
        # starts with "checkpoint"
        directories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        checkpoint_directories = [d for d in directories if d.startswith("checkpoint")]

        api.upload_folder(
            folder_path=os.path.join(output_dir, checkpoint_directories[0]),
            repo_id=f"{username}/{output_repo}",
            repo_type="model",
        )


if __name__ == "__main__":
    main()
