# ======================================================================
# Author: Austin Tao (modified from drug_experiment/merge_eval.py)
#
# script to determine why we are seeing higher than average test loss values with our merge
#
#
#======================================================================

import torch
import numpy as np
import pandas as pd
import evaluate
from typing import Any, Tuple, Dict  # for type annotations
from torch.nn.functional import softmax
from datasets import load_dataset,Dataset
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, PreTrainedTokenizerBase
from sklearn.metrics import accuracy_score


def eval_with_probs(trainer: Trainer,
                              test_dataset: Dataset,
                              validation_dataset:Dataset,
                              output_dir: str = "test_loss_exp") -> None:
    """
    Evaluate the model on test and validation datasets using the provided Trainer object,
    compute probabilities, and save the results along with accuracy to CSV files.

    Args:
        - trainer (Trainer): The Hugging Face Trainer object.
        - test_dataset (datasets.Dataset): The dataset for testing.
        - validation_dataset (datasets.Dataset): The dataset for validation.
        - output_dir (str): Directory path where the results will be saved.

    Returns:
        - None. Outputs 
    """
    # define datasets
    datasets = {'test': test_dataset, 'validation': validation_dataset}

    # go through both test and val sets
    for name, dataset in datasets.items():
        # Predict using the trainer
        predictions = trainer.predict(dataset)
        
        # Extract logits and convert to probabilities
        logits = predictions.predictions
        probabilities = softmax(torch.tensor(logits), dim=-1).numpy()

        # add the true labels
        labels = dataset["label"]
        df = pd.DataFrame(probabilities)
        df["true_label"] = dataset["label"]
        
        # Save probabilities to a CSV file
        probabilities_file = f"{output_dir}/{name}_probabilities.csv"
        df.to_csv(probabilities_file, index = False)
        print(f"Saved {name} probabilities to {probabilities_file}")
        
        # Calculate and print accuracy if labels are present
        if hasattr(predictions, 'label_ids') and predictions.label_ids is not None:
            true_labels = predictions.label_ids
            predicted_labels = logits.argmax(axis=-1)
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f"{name.capitalize()} Accuracy: {accuracy}")
            # Save accuracy to a file
            with open(f"{output_dir}/{name}_accuracy.txt", 'w') as f:
                f.write(f"{name.capitalize()} Accuracy: {accuracy}\n")



def train_test_val_split(dataset, tokenizer, prop_eval: float = 0.2):
    """
    Method to split dataset into training, testing, and validation (and tokenized)

    Args:
        - dataset:
        - tokenizer: 
        - prop_eval: proportion WITHIN testing data to set aside for validation

    Returns:
        - Tuple containing training, testing, and validation datasets (tokenized)
    """
    # separate into training and eval
    train_data = dataset["train"]
    eval_data = dataset["test"]

    # separagte based on reviews and ratings
    reviews_train = train_data['review']
    ratings_train = train_data['rating']
    reviews_eval = eval_data['review']
    ratings_eval = eval_data['rating']

    # create datasets from dictionaries
    drug_data_train = {'label': [int(rating-1) for rating in ratings_train],'text': reviews_train}
    drug_data_train = Dataset.from_dict(drug_data_train)
    drug_data_eval = {'label': [int(rating-1) for rating in ratings_eval],'text': reviews_eval}
    drug_data_eval = Dataset.from_dict(drug_data_eval)

    # tokenization
    tokenized_drug_train = drug_data_train.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),
                                               batched=True)
    tokenized_drug_test = drug_data_eval.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),
                                             batched=True)
    eval_dataset = tokenized_drug_test.shuffle(seed=42)

    # Define the size of the validation set (e.g., 20% of the total data)
    validation_size = int(0.2 * len(eval_dataset))
    indices = list(range(len(eval_dataset)))
    test_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]

    # Split the shuffled training dataset into training and validation sets
    test_dataset = eval_dataset.select(test_indices)
    validation_dataset = eval_dataset.select(validation_indices)

    # rename training set for continuity
    train_dataset = tokenized_drug_train

    return train_dataset, test_dataset, validation_dataset



def write_results(test_results: Any, 
                  val_results: Any, 
                  file_path: str = "model_results.txt") -> None:
    """
    Write the results of our model, including the current date and time, test results, and validation results,
    to a specified file. The file will be appended to if it already exists.
    
    Args:
        - test_results (Any): The results from testing the model. This can be any data type that is compatible
                              with string formatting, typically str or a type that implements __str__.
        - val_results (Any): The results from validating the model. This can be any data type that is compatible
                             with string formatting.
        - file_path (str, optional): The path to the file where results will be written. Defaults to "model_results.txt".
    
    Returns:
        - None: This function does not return any values.
    """
    # Hold the current date and time information
    time_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Generate result string
    result_str = "Date and Time: {}\nTest Results: {}\nValidation Results: {}".format(time_now, test_results, val_results)

    # Write to file
    with open(file_path, "a") as file:
        file.write(result_str)


    # def write_results(test_results, val_results, file_path: str = "model_results.txt"):
    #     """ 
    #     write the results of our model to a file
    # 
    # 
    #     """
    #     # hold the current date and time information
    #     time_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # 
    #     # generate result string
    #     result_str = "Date and Time: {}\nTest Results: {}\nValidation Results: {}".format(time_now,
    #                                                                                       test_results,
    #                                                                                       val_results)
    #     # write to file
    #     with open(file_path, "a") as file:
    #         file.write(result_str)

    # f = open("accuracy_merge_drug_data_dare_linear.txt", "a")
    # f.write(f"DARE Ties merge validation results : {validation_result}\n")
    # f.write(f"DARE Ties test results : {test_result}\n\n")
    # f.close()
def preprocess_function(examples: Dict[str, Any],
                        tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    Preprocesses the text data by tokenizing the input text using the specified tokenizer.
    
    Args:
        - examples (Dict[str, Any]): dict containing the text examples with keys corresponding to feature names and values.
        - tokenizer (PreTrainedTokenizerBase): The tokenizer instance from the transformers library

    Returns:
        - Dict[str, Any]: A dictionary containing the tokenized text data.
    """
    return tokenizer(examples["text"], truncation=True,max_length=1024)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes the accuracy of model predictions using the 'accuracy' metric from the Hugging Face datasets library.
    
    Args:
        - eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing two elements:
            1. predictions (np.ndarray): The raw model output logits or probabilities.
            2. labels (np.ndarray): The true labels against which to evaluate the predictions.
    
    Returns:
        - Dict[str, float]: A dictionary containing the computed accuracy metric.
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main():
    # load tokenizer
    #tokenizer = AutoTokenizer.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_ties")
    tokenizer = AutoTokenizer.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_linear")
    model = AutoModelForSequenceClassification.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_linear")

    # load dataset
    dataset = load_dataset("lewtun/drug-reviews")
    
    # split into train-test-val
    training_dataset, test_dataset, validation_dataset = train_test_val_split(dataset, tokenizer)

    # Load model directly
    #model = AutoModelForSequenceClassification.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_ties")
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train model
    trainer = Trainer(
        model=model,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # print results
    # validation_result = trainer.evaluate(eval_dataset=validation_dataset)
    # test_result = trainer.evaluate(eval_dataset=test_dataset)
    # write_results(test_result, validation_result)

    # get probabilities
    eval_with_probs(trainer, test_dataset, validation_dataset)


if __name__ == "__main__":
    main()



