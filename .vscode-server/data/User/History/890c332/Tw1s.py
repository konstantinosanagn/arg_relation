import json
import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

# Define the filename
filename = "/home/parklab/data/debugging_dataset/debugging.jsonlist"

# Check if the file already exists
if not os.path.exists(filename):
    raise FileNotFoundError(f"The file {filename} does not exist. Please provide a valid file path.")

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    print("Data loaded successfully.")
    return data

def create_dataset(data):
    dataset = []
    for comment in data:
        comment_id = comment['id']
        for prop in comment['propositions']:
            proposition_id = prop['id']
            text = prop['text']
            for s_prop in comment['propositions']:
                if prop['id'] != s_prop['id']:
                    dataset.append({
                        'comment_id': comment_id,
                        'proposition_id': proposition_id,
                        'sup_proposition_id': s_prop['id'],
                        'text': text,
                        'sup_text': s_prop['text'],
                        'actual_relation': s_prop['id'] in (prop['reasons'] or [])
                    })
    print("Dataset created successfully.")
    return pd.DataFrame(dataset)

def generate_proposition_pairs(comment: Comment) -> List[Tuple[Proposition, List[Proposition]]]:
    """
    Generate pairs of the ith proposition with the five propositions before it
    and the five propositions after it.

    Args:
        comment (Comment): The comment object containing propositions.

    Returns:
        List[Tuple[Proposition, List[Proposition]]]: A list of tuples, where each tuple contains
        a proposition and a list of up to 10 surrounding propositions.
    """
    pairs = []
    propositions = comment.propositions
    for i in range(len(propositions)):
        surrounding = propositions[max(0, i-5):i] + propositions[i+1:i+6]
        pairs.append((propositions[i], surrounding))
    return pairs

# Load Mistral v0.3 model and tokenizer from Hugging Face
print("Loading Mistral v0.3 model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Mistral v0.3 model loaded successfully and moved to device.")

def generate_response(prompt: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

from src.lib.link_prediction.contextual_assistant_response_parser import SupportResponseParser

def predict_relations(df, response_parser):
    for index, row in df.iterrows():
        prompt = (
            "Given the following two propositions:\n"
            f"1. {row['text']}\n"
            f"2. {row['sup_text']}\n"
            "Is the second proposition a supporting reason for the first proposition?\n"
            "Answer with 'yes' or 'no'."
        )
        response = generate_response(prompt)
        parsed_response = response_parser.get_parsed_response(response)
        df.at[index, 'predicted_relation'] = parsed_response
        
        if index % 10 == 0:
            print(f"Processed {index} proposition pairs.")
    print("Prediction of relations completed.")
    return df

def get_proposition_text(df, comment_id, proposition_id):
    proposition = df[(df['comment_id'] == comment_id) & (df['proposition_id'] == proposition_id)]
    if not proposition.empty:
        return proposition.iloc[0]['text']
    return None

# Compute precision, recall, F1 score, and accuracy
def calculate_metrics(df):
    y_true = df['actual_relation']
    y_pred = df['predicted_relation'] == 'yes'
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, accuracy, recall, f1

# Read data from uploaded file
data = read_jsonl(filename)

# Create dataset
df = create_dataset(data)

# Initialize the response parser
response_parser = SupportResponseParser()

# Predict relations
df = predict_relations(df, response_parser)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# Retrieve specific proposition text
text = get_proposition_text(df, comment_id=194, proposition_id=0)
print(text)

# Calculate and print metrics
precision, accuracy, recall, f1 = calculate_metrics(df)
print(f'Precision: {precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
