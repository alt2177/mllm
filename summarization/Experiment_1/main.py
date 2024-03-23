from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import AutoTokenizer,DataCollatorForSeq2Seq,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,GPT2LMHeadModel
from datasets import load_dataset
import evaluate
import numpy as np
import torch

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge
rouge = evaluate.load("rouge")

prefix = "summarize: "

def preprocess_function(examples,tokenizer):
    inputs = [prefix + doc + " TL;DR" for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True,padding="max_length")
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True,padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def compute_metrics(eval_pred,tokenizer):
    #tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small",use_auth_token = "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" )
    predictions, labels = eval_pred
    print(f"Predictions before : {predictions}")
    print(f"Labels : {labels}")
    #predictions = predictions[0][0]
    #print(f"Predictions : {predictions}")
    #print(f"Labels : {labels}")
    #predictions = [item for sublist in predictions for item in sublist]
    #labels = [item for sublist in labels for item in sublist]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

def main():
    # set our collective token and HF info
    access_token = "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" 
    username = "mllm-dev"
    # repo that will be made on huggingface
    output_repo = "bill_sum_experiment_1"

    # load and tokenize data
    dataset = load_dataset("billsum", split="ca_test")
    dataset = dataset.train_test_split(test_size=0.2,seed=42)
    checkpoint = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_billsum = dataset.map(lambda examples:preprocess_function(examples,tokenizer),batched=True)
    #print(f"test : {tokenized_billsum['train']['labels'][0]}")    
    # pad tokens
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=checkpoint) 

    # create model
    model = GPT2LMHeadModel.from_pretrained(checkpoint)


    # model.config.pad_token_id = tokenizer.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))

    # small_train_dataset = tokenized_yelp["train"].shuffle(seed=42)
    # small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42)

    # create the repo before we try to push the model to huggingface

    HfFolder.save_token(access_token)
    api = HfApi()
    user = api.whoami(token=access_token)
    try:
        create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "bill_sum_finetune_test_gpt2"
    # training loop
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x : compute_metrics(x,tokenizer),
    )

    print('************TRAINING STARTED*************')
    trainer.train()
    print('*************TRAINING COMPLETED**************')
   # trainer.evaluate()

if __name__ == "__main__":
    main()


