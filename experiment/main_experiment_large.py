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
    #model_name = "microsoft/phi-1"
    model_name = "openai-community/gpt2-xl"

    # load and tokenize data
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 
    
    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            use_auth_token = access_token
        )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # repo that will be made on huggingface
    output_repo = f"gpt_f_experiment_large"

    # datasets train/test
    train_dataset = tokenized_yelp["train"]
    eval_dataset = tokenized_yelp["test"].shuffle(seed=42)

    # Define the size of the validation set (e.g., 20% of the total data)
    validation_size = int(0.2 * len(eval_dataset))
    indices = list(range(len(eval_dataset)))
    test_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]

    # Split the shuffled training dataset into training and validation sets
    test_dataset = eval_dataset.select(test_indices)
    validation_dataset = eval_dataset.select(validation_indices)
    
    # create the repo before we try to push the model to huggingface
    HfFolder.save_token(access_token)
    api = HfApi()
    user = api.whoami(token=access_token)
    try:
        create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "sean_test_out_large"
    # training loop
    training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
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
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    print('************TRAINING STARTED*************')
    result = trainer.train()
    print('*************TRAINING COMPLETED**************')  
    
    # Check the amount of GPU memory currently allocated
    allocated_memory = torch.cuda.memory_allocated()
    print("Allocated GPU memory:", allocated_memory / 1024**2, "MB")

    # Check the peak GPU memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    print("Peak GPU memory usage:", peak_memory / 1024**2, "MB")

    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    validation_result = trainer.evaluate(eval_dataset = validation_dataset)
    
    # Save the accuracy of each model for later comparison
    f = open("accuracy_large.txt", "a")
    f.write(f"Fine tune model {m} : {result}\n")
    f.write(f"Fine tine model {m} validation result : {validation_result}\n")
    f.write(f"Fine tune large model accuracy : {eval_result}\n")
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
