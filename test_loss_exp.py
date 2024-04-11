#
#
#
#

from datasets import load_dataset,Dataset
from datetime import datetime
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

def train_test_val_split(dataset, tokenizer, prop_eval: float = 0.2):
    """
    Method to split dataset into training, testing, and validation (and tokenized)

    Args:
        dataset:
        tokenizer: 
        prop_eval: proportion WITHIN testing data to set aside for validation

    Returns:
        Tuple containing training, testing, and validation datasets (tokenized)
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
    tokenized_drug_train = drug_data_train.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),batched=True)
    tokenized_drug_test = drug_data_eval.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),batched=True)
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


def write_results(test_results, val_results, file_path: str = "model_results.txt"):
    """ 
    write the results of our model to a file


    """
    # hold the current date and time information
    time_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # generate result string
    result_str = "Date and Time: {}\nTest Results: {}\nValidation Results: {}".format(time_now,
                                                                                      test_results,
                                                                                      val_results)
    # write to file
    with open(file_path, "a") as file:
        file.write(result_str)

    # f = open("accuracy_merge_drug_data_dare_linear.txt", "a")
    # f.write(f"DARE Ties merge validation results : {validation_result}\n")
    # f.write(f"DARE Ties test results : {test_result}\n\n")
    # f.close()


def preprocess_function(examples, tokenizer):
    """

    """
    return tokenizer(examples["text"], truncation=True,max_length=1024)


def compute_metrics(eval_pred):
    """

    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_ties")

    # load dataset
    dataset = load_dataset("lewtun/drug-reviews")
    
    # split into train-test-val
    training_dataset, test_dataset, validation_dataset = train_test_val_split(dataset, tokenizer)

    # Load model directly
    model = AutoModelForSequenceClassification.from_pretrained("mllm-dev/gpt2_m_experiment_drug_data_ties")
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    accuracy = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # print results
    validation_result = trainer.evaluate(eval_dataset=validation_dataset)
    test_result = trainer.evaluate(eval_dataset=test_dataset)
    write_results(test_result, validation_result)


    #print("Merge validation:", validation_result)

if __name__ == "__main__":
    main()



# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
# model = AutoModelForSequenceClassification.from_pretrained(
#     "mllm-dev/gpt2_m_experiment_drug_data_dare_linear", num_labels=10
# )





#small_eval_dataset = tokenized_yelp["test"]










