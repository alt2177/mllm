<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
# MLLM Class source file
# 

import os
<<<<<<< HEAD
=======
# Class for MLLM, which we can create instances of to run tests or otherwise
#
#

>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
# MLLM Class source file
# 

import os
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> main


class MLLM:

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> main
    # HuggingFace access token
    access_token = "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"

    # specify output directory
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    output_dir = "test_MLLM"
=======
    output_dir = "test"
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
    output_dir = "test_MLLM"
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
    output_dir = "test_MLLM"
>>>>>>> main

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
        push_to_hub=False,
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
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

<<<<<<< HEAD
<<<<<<< HEAD
=======
        self.model = self.create_model()
        self.dataset = self.load_dataset() 
=======
        self.model = create_model(self)
        self.dataset = load_dataset(self) 
>>>>>>> f97b2ef44 (began testing MLLM.py)
        self.result = None
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main

    
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
<<<<<<< HEAD
<<<<<<< HEAD
        elif hf_dataset_name:
=======
        elif dataset_name:
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
        elif hf_dataset_name:
>>>>>>> main
            dataset = load_dataset(hf_dataset_name)
        else:
            raise ValueError("Either dataset_name or local_path must be provided.")
        
        return dataset


    def tokenize_data(self):
        """

        """
        assert self.dataset is not None, "Dataset Missing"
    
        # create tokenizer and tokenize
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = self.access_token)
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_data = self.dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
<<<<<<< HEAD
<<<<<<< HEAD
=======
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_data = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main

        # set data collator with tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer = tokenizer) 
        self.dataset = tokenized_data

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def train_test_val_split(self, train_size : int = 1000, test_size : int = 1000, val_size : int = None):
=======
    def train_test_val_split(self, train_size : int, test_size : int, val_size : int = None):
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
    def train_test_val_split(self, train_size : int = 1000, test_size : int = 1000, val_size : int = None):
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
    def train_test_val_split(self, train_size : int = 1000, test_size : int = 1000, val_size : int = None):
>>>>>>> main
        """

        """
        train_dataset = self.dataset["train"].shuffle(seed=42).select(range(train_size))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main
        test_dataset = self.dataset["test"].shuffle(seed=4).select(range(test_size))
        
        # create validation set if we ask for one
        if val_size:
            val_dataset = self.dataset["train"].shuffle(seed=420).select(range(val_size))
<<<<<<< HEAD
=======
        test_dataset = self.dataset["test"].shuffle(seed=42).select(range(test_size))
        
        # create validation set if we ask for one
        if val_size:
            val_dataset = self.dataset["train"].shuffle(seed=42).select(range(val_size))
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
        test_dataset = self.dataset["test"].shuffle(seed=4).select(range(test_size))
        
        # create validation set if we ask for one
        if val_size:
            val_dataset = self.dataset["train"].shuffle(seed=420).select(range(val_size))
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
            use_auth_token = self.access_token
        )
        model.config.pad_token_id = self.tokenizer.eos_token_id
        model.resize_token_embeddings(len(self.tokenizer))
<<<<<<< HEAD
<<<<<<< HEAD
=======
            use_auth_token = access_token
=======
            use_auth_token = cls.access_token
>>>>>>> f97b2ef44 (began testing MLLM.py)
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
        return model


    def train(self):

        trainer = Trainer(
            model=self.model,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
<<<<<<< HEAD
<<<<<<< HEAD
=======
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> b83eb4ba6 (debugged MLLM)
=======
>>>>>>> main
        )

        self.result = trainer.train()
        print(self.result)

    def write_results(self, file_name: str = "results.txt"):
        with open(file_name, "w") as file:
            results = "".join(self.result.to_tuple())
            file.write(results)





<<<<<<< HEAD
<<<<<<< HEAD
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
=======
>>>>>>> be4e74db6 (began transitioning code into package structure)
=======
>>>>>>> main




