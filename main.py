import mllm

def main():
<<<<<<< HEAD
<<<<<<< HEAD
    # set our collective token
    access_token = "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" 
    username = "mllm-dev"
    output_repo = "yelp_finetuned_sbatch_upload_test"

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
    small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(10))
 
    output_dir = "yelp_finetune_gpt2_test"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
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
<<<<<<< HEAD
<<<<<<< HEAD
        hub_token=access_tokens,
=======
=======
	    hub_model_id=f"{username}/{output_repo}",
>>>>>>> f97b2ef44 (began testing MLLM.py)
=======
	    hub_model_id=f"{username}/{output_repo}",
>>>>>>> main
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
<<<<<<< HEAD
        compute_metrics=compute_metrics,
<<<<<<< HEAD
        hub_token=access_tokens,
=======
#        hub_token = access_token
>>>>>>> e8a460892 (model trained)
=======
        compute_metrics=compute_metrics
>>>>>>> f97b2ef44 (began testing MLLM.py)
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
	my_mllm = mllm.MLLM()
	my_mllm.train()
	my_mllm.write_results()
	print("DONE!")
>>>>>>> 2b79c839a (confirmed MLLM functional on Lambda)
=======
	print("================= BEGIN! =================")
	my_mllm = mllm.MLLM()
	my_mllm.train()
	my_mllm.write_results()
	print("================= DONE! =================")
>>>>>>> 898b93796 (fixed results printing)


if __name__ == "__main__":
	main()
