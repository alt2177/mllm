<<<<<<< HEAD
# MLLM Class source file
# 

import os
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
=======
"""
Class file for the actual MLLM we want to use
"""

import torch
import yaml
import mergekit as mk
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import os



>>>>>>> f522bc7a8 (began trying to use mergekit in our data pipeline)


class MLLM:

<<<<<<< HEAD
    # HuggingFace access token
    access_token = "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"

    # specify output directory
    output_dir = "test_MLLM"

    # training arguments
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
        hub_token=access_token
    )

    @classmethod
    def compute_metrics(cls, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    def __init__(self):
        """

        """
        # set dataset and tokenize
        self.dataset = self.load_dataset() 
        self.tokenize_data()

        # split data
        self.train_dataset, self.test_dataset = self.train_test_val_split()

        # create our model
        self.model = self.create_model()

        # initialize result variables
        self.result: Object
        self.tokenizer: AutoTokenizer 


    
    def load_dataset(self, hf_dataset_name: str = "yelp_review_full" , local_path: str = None):
        """
        Loads a dataset from Hugging Face or a local file.

        Parameters:
            - hf_dataset_name: str (optional) - Name of the dataset to load from Hugging Face.
            - local_path: str (optional) - Path to the local dataset file.

        Returns:
            - A dataset object loaded from Hugging Face or local file.
        """
        # if we specify a local path, that takes precedence
        if local_path:
            if os.path.exists(local_path):
                # Assuming the local dataset is a CSV file for simplicity.
                # Adjust the loading mechanism based on your dataset format.
                dataset = load_dataset('csv', data_files=local_path)
            else:
                raise FileNotFoundError(f"The specified path does not exist: {local_path}")
        # otherwise, load from HF
        elif hf_dataset_name:
            dataset = load_dataset(hf_dataset_name)
        else:
            raise ValueError("Either dataset_name or local_path must be provided.")
        
        return dataset


    def tokenize_data(self):
        """

        """
        assert self.dataset is not None, "Dataset Missing"
    
        # create tokenizer and tokenize
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = self.access_token)
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_data = self.dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

        # set data collator with tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer = tokenizer) 
        self.dataset = tokenized_data

    def train_test_val_split(self, train_size : int = 1000, test_size : int = 1000, val_size : int = None):
        """

        """
        train_dataset = self.dataset["train"].shuffle(seed=42).select(range(train_size))
        test_dataset = self.dataset["test"].shuffle(seed=4).select(range(test_size))
        
        # create validation set if we ask for one
        if val_size:
            val_dataset = self.dataset["train"].shuffle(seed=420).select(range(val_size))
            return train_dataset, test_dataset, val_dataset
        else:
            return train_dataset, test_dataset


    def create_model(self):
        """
        Create model 
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            "openai-community/gpt2",
            num_labels = 5,
            use_auth_token = self.access_token
        )
        model.config.pad_token_id = self.tokenizer.eos_token_id
        model.resize_token_embeddings(len(self.tokenizer))
        return model


    def train(self, train_dataset, test_dataset):

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.result = trainer.train()
        print(result)







=======

    def __init__(self) -> None:
        """
        constructor
        """
        # variables necessary for mergekit
        OUTPUT_PATH = "./merged"  # folder to store the result in
        LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
        CONFIG_YML = "../tests/ultra_llm_merged.yml"  # merge configuration file
        COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
        LAZY_UNPICKLE = False  # experimental low-memory model loader
        LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

        # self.set_paths()
        with open(CONFIG_YML, "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
        run_merge(
             merge_config,
             out_path=OUTPUT_PATH,
             options=MergeOptions(
                 lora_merge_cache=LORA_MERGE_CACHE,
                 cuda=torch.cuda.is_available(),
                 copy_tokenizer=COPY_TOKENIZER,
                 lazy_unpickle=LAZY_UNPICKLE,
                 low_cpu_memory=LOW_CPU_MEMORY,
             ),
         )
        print("Done!")

        pass

    def set_paths(self) -> None:
        """
        Make sure we have the right path to access yaml files
        """
        # create our full path
        full_path = os.path.join(os.getcwd(), "mergekit-yaml")

        # set path we want to change to
        new_directory = "venv/bin/"

        # check if we have mergekit in our cwd
        if not os.path.exists(full_path):
            try:
                os.chdir(new_directory)
                print(f"Directory changed to: {os.getcwd()}")
            except FileNotFoundError:
                print(f"Directory not found: {new_directory}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print(f"{file_or_directory} exists in the current directory. No change made.")
        print(os.getcwd())
        pass
>>>>>>> f522bc7a8 (began trying to use mergekit in our data pipeline)




