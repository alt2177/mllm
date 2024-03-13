from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, GPT2Config, Trainer, TrainingArguments 
from datasets import load_dataset
import evaluate
import numpy as np
import torch

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge
def preprocess_function(examples, tokenizer):
	questions = [q.strip()[2:-3] for q in examples['question']]
	answers = [a.strip()[2:-3] for a in examples['answer']]
	text = ["Question: " + q + "    Answer: " + a for q, a in zip(questions, answers)]
	return tokenizer(text, padding='max_length', truncation=True)

#def preprocess_function(examples, tokenizer):
#	questions = [q.strip() for q in examples['question']]
#	answers = [a.strip() for a in examples['answer']]
#	text = [q + " " + a for q, a in zip(questions, answers)]
#	return tokenizer(text, padding='max_length', truncation=True)

def main():
	# set our collective token and HF info
	access_token = "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"
	username = "mllm-dev"
	# repo that will be made on huggingface
	output_repo = "gen_test_5"

	#model_id = "google/gemma-2b"
	model_id = "openai-community/gpt2"

	# load and tokenize data
	dataset = load_dataset("math_dataset", "arithmetic__add_or_sub")
	tokenizer = GPT2Tokenizer.from_pretrained(model_id, use_auth_token=access_token)
	tokenizer.pad_token = tokenizer.eos_token

	configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
	model = GPT2LMHeadModel.from_pretrained(model_id, config=configuration, use_auth_token=access_token)

	#text = "Total of 0.06 and -1977321735."
	#inputs = tokenizer(text, return_tensors="pt")
	#outputs = model.generate(**inputs, max_new_tokens=20)
	#print('Question: ', text)
	#print('Answer: ', tokenizer.decode(outputs[0], skip_special_tokens=True))

	#return
	#small_train = dataset['train'].select(range(100000))
	#dataset['train'] = small_train
	#print('train ', dataset['train'][0]['question'][2:-3])
	tokenized_data = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
	# pad tokens
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	model.config.pad_token_id = tokenizer.eos_token_id
	model.resize_token_embeddings(len(tokenizer))

	small_train_dataset = tokenized_data["train"].shuffle(seed=42)
	small_eval_dataset = tokenized_data["test"].shuffle(seed=42)

	# create the repo before we try to push the model to huggingface
	HfFolder.save_token(access_token)
	api = HfApi()
	user = api.whoami(token=access_token)
	try:
		create_repo(f"{username}/{output_repo}", repo_type="model")
	except:
		print('error creating repo for model. it probably already exists')


	output_dir = "gen_test"
	# training loop
	training_args = TrainingArguments(
		output_dir=output_dir,
		learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
		#gradient_accumulation_steps=4,   #want to explore what this is
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
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
		data_collator=data_collator
	)
	
	print('************TRAINING STARTED*************')
	trainer.train()
	print('*************TRAINING COMPLETED**************')
	#trainer.evaluate()

	print('UPLOADING')

	api.upload_folder(
		folder_path=f"./{output_dir}",
		repo_id=f"{username}/{output_repo}",
		repo_type="model"
	)
	print('uploading done!')
if __name__ == "__main__":
	main()
