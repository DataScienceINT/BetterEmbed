import json
import random

def split_dataset(dataset, train_ratio=0.8):
    """
    Split a dataset into training and test sets.
    
    :param dataset: A dictionary containing the dataset.
    :param train_ratio: The ratio of data to use for training (default: 80%).
    :return: Two dictionaries (train_set, test_set).
    """
    # Convert dataset into a list of tuples (id, content)
    dataset_items = list(dataset.items())
    
    # Shuffle the dataset
    random.shuffle(dataset_items)
    
    # Calculate the split index
    split_index = int(len(dataset_items) * train_ratio)
    
    # Split into training and test sets
    train_items = dataset_items[:split_index]
    test_items = dataset_items[split_index:]
    
    # Convert back to dictionaries
    train_set = dict(train_items)
    test_set = dict(test_items)
    
    return train_set, test_set

def save_json(data, filename):
    """
    Save a dictionary as a JSON file.
    
    :param data: The data to save.
    :param filename: The name of the JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def load_bioasq(file_path):
    """
    Load questions and answers from the BioASQ JSON file, transforming into the format:
    {"id": {"question": "question text", "answers": ["answer1 text", "answer2 text", ...]}}
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    transformed_data = {}
    
    # Iterate over each question
    for question_data in data["questions"]:
        question_id = question_data["id"]  # Extract the question ID
        question_text = question_data["body"]  # Extract the question text
        answers = [snippet["text"] for snippet in question_data["snippets"]]  # Extract answers from snippets
        
        # Store the question and answers in the dictionary under the question ID
        transformed_data[question_id] = {
            "question": question_text,
            "answers": answers
        }
    
    return transformed_data

def load_pubmedqa(file_path):
    """
    Load questions and answers from the PubmedQA JSON file, transforming into the format:
    {"id": {"question": "question text", "answers": ["long answer text"]}}
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    transformed_data = {}
    
    # Iterate over each entry in the dataset
    for entry_id, entry_data in data.items():
        question_text = entry_data.get("QUESTION", "")  # Extract the question text
        long_answer = entry_data.get("LONG_ANSWER", "")  # Extract the long answer
        
        # Wrap the single long answer in a list to maintain the format consistency
        transformed_data[entry_id] = {
            "question": question_text,
            "answers": [long_answer]  # Single long answer as a list
        }
    
    return transformed_data

if __name__ == "__main__":
    # Load the BioASQ dataset
    dataset = load_bioasq('bioasq12b.json')
    pubmedqa = load_pubmedqa('ori_pqal.json')
    
    # Split the dataset into training (80%) and test (20%)
    train_set, test_set = split_dataset(dataset, train_ratio=0.8)
    
    # Save the training and test sets to JSON files
    save_json(train_set, 'bioasq_train_set.json')
    save_json(test_set, 'bioasq_test_set.json')
    save_json(pubmedqa, 'pubmedqa_test_set.json')
    
    print("Training and test sets have been saved!")
