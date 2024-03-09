from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import GemmaTokenizer,AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig
import evaluate
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge

def compute_metrics(eval_pred):
	generated_answers = [pred.strip().decode("utf-8") for pred in eval_pred.predictions]
	reference_answers = [ref.strip().decode("utf-8") for ref in eval_pred.label_ids]

	# Compute BLEU score
	bleu_score = corpus_bleu([[ref] for ref in reference_answers], generated_answers)

	return {"bleu_score": bleu_score}
# Assume each example in the dataset has "question" and "answer" fields
def preprocess_function(examples, tokenizer):
	questions = [q for q in examples['question']]  # Decoding might be necessary if the text is in bytes
	inputs = tokenizer(questions, padding="max_length", truncation=True, max_length=512)

	answers = [a for a in examples['answer']]
	labels = tokenizer(answers, padding="max_length", truncation=True, max_length=512).input_ids
	inputs["labels"] = labels
	
	return inputs

def main():
	# set our collective token and HF info
	access_token = "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"
	username = "mllm-dev"
	# repo that will be made on huggingface
	output_repo = "gemma_test"

	lora_config = LoraConfig(
		r=8,
		target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
		task_type="CAUSAL_LM",
	)

	model_id = "google/gemma-2b"

	# load and tokenize data
	dataset = load_dataset("math_dataset", "arithmetic__add_or_sub")
	tokenizer = GemmaTokenizer.from_pretrained(model_id, use_auth_token=access_token)
	model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token)
	tokenizer.pad_token = tokenizer.eos_token

	#text = "Total of 0.06 and -1977321735."
	#inputs = tokenizer(text, return_tensors="pt")
	#outputs = model.generate(**inputs, max_new_tokens=20)
	#print('Question: ', text)
	#print('Answer: ', tokenizer.decode(outputs[0], skip_special_tokens=True))

	#return
	small_train = dataset['train'].select(range(1000))
	dataset['train'] = small_train
	tokenized_data = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
	# pad tokens
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	model.config.pad_token_id = tokenizer.eos_token_id
	model.resize_token_embeddings(len(tokenizer))

	small_train_dataset = tokenized_data["train"].shuffle(seed=42).select(range(100))
	small_eval_dataset = tokenized_data["test"].shuffle(seed=42).select(range(100))

	# create the repo before we try to push the model to huggingface
	HfFolder.save_token(access_token)
	api = HfApi()
	user = api.whoami(token=access_token)
	try:
		create_repo(f"{username}/{output_repo}", repo_type="model")
	except:
		print('error creating repo for model. it probably already exists')


	output_dir = "gem_test"
	# training loop
	training_args = TrainingArguments(
		output_dir=output_dir,
		learning_rate=2e-5,    #read that with larger batch size we can increase learning rate
		#gradient_accumulation_steps=4,   #want to explore what this is
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
		push_to_hub=True,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=small_train_dataset,
		eval_dataset=small_eval_dataset,
		tokenizer=tokenizer,
		#data_collator=data_collator,
		compute_metrics=compute_metrics,
	)
	print('************TRAINING STARTED*************')
	trainer.train()
	print('*************TRAINING COMPLETED**************')
	#trainer.evaluate()

if __name__ == "__main__":
	main()
