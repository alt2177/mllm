# Class for MLLM, which we can create instances of to run tests or otherwise
#
#

from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np


class MLLM:

    # HuggingFace access token
    access_token = "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"

    # specify output directory
    output_dir = "test"

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
        self.model = self.create_model()
        self.dataset = self.load_dataset() 
        self.result = None

    
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
        elif dataset_name:
            dataset = load_dataset(hf_dataset_name)
        else:
            raise ValueError("Either dataset_name or local_path must be provided.")
        
        return dataset


    def tokenize_data(self):
        """

        """
        assert self.dataset is not None, "Dataset Missing"
    
        # create tokenizer and tokenize
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_data = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

        # set data collator with tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer = tokenizer) 
        self.dataset = tokenized_data

    def train_test_val_split(self, train_size : int, test_size : int, val_size : int = None):
        """

        """
        train_dataset = self.dataset["train"].shuffle(seed=42).select(range(train_size))
        test_dataset = self.dataset["test"].shuffle(seed=42).select(range(test_size))
        
        # create validation set if we ask for one
        if val_size:
            val_dataset = self.dataset["train"].shuffle(seed=42).select(range(val_size))
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
            use_auth_token = access_token
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        return model


    def train(self, train_dataset, test_dataset):

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        self.result = trainer.train()
        print(result)











