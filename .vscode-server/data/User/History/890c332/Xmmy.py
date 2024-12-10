import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from typing import List, Tuple
from parsers import BaseResponseParser, SupportResponseParser

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

def generate_proposition_pairs(comment) -> List[Tuple[dict, List[dict]]]:
    pairs = []
    propositions = comment['propositions']
    for i in range(len(propositions)):
        surrounding = propositions[max(0, i-5):i] + propositions[i+1:i+6]
        pairs.append((propositions[i], surrounding))
    return pairs

def create_reasons_map(comments) -> dict:
    reasons_map = {}
    for comment in comments:
        comment_id = comment['id']
        for prop in comment['propositions']:
            proposition_id = prop['id']
            reasons = prop['reasons'] if prop['reasons'] else []
            all_reasons = []
            for reason in reasons:
                if '_' in reason:
                    start, end = map(int, reason.split('_'))
                    all_reasons.extend(range(start, end + 1))
                else:
                    all_reasons.append(int(reason))
            reasons_map[(comment_id, proposition_id)] = all_reasons
    return reasons_map

def create_dataset(data):
    dataset = []
    for comment in data:
        comment_id = comment['id']
        pairs = generate_proposition_pairs(comment)
        for prop, surrounding in pairs:
            for s_prop in surrounding:
                dataset.append({
                    'comment_id': comment_id,
                    'proposition_id': prop['id'],
                    'sup_proposition_id': s_prop['id'],
                    'text': prop['text'],
                    'sup_text': s_prop['text'],
                    'predicted_relation': None
                })
    print("Dataset created successfully.")
    return pd.DataFrame(dataset)

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

def create_prompt(row, response_parser):
    prompt = (
        "Task Description:\n\n"
        "Objective: Determine whether the second sentence supports the first sentence.\n\n"
        "Instructions:\n"
        "1. Read the following two sentences carefully:\n"
        f"   - Sentence 1: {row['text']}\n"
        f"   - Sentence 2: {row['sup_text']}\n\n"
        "2. Question: Does Sentence 2 support Sentence 1?\n\n"
        "3. Response Format: With no other words, answer with 'yes' or 'no'.\n\n"
        "Example:\n\n"
        "Sentence 1: Recently, courts have held that debt collectors can escape 1692i's venue provisions entirely by pursuing debt collection through arbitration instead.\n"
        "Sentence 2: As the NAF studies reflect, arbitration has not proven a satisfactory alternative.\n"
        "Question: Does Sentence 2 support Sentence 1?\n"
        "Answer: yes\n\n"
        "Explanation (optional): The second sentence suggests that arbitration is not a satisfactory alternative for debt collection. "
        "This implies that debt collectors may be pursuing debt collection through arbitration as an alternative to the venue provisions in 1692i. "
        "Therefore, the second sentence supports the first sentence.\n\n"
        "Now, your task:\n\n"
        f"1. Sentence 1: {row['text']}\n"
        f"2. Sentence 2: {row['sup_text']}\n\n"
        "Question: Does Sentence 2 support Sentence 1?\n"
        f"{response_parser.answer_format.format(response_parser.answer_token)}"  # Incorporate the answer token and format
    )
    return prompt


def get_response(prompt):
    response = generate_response(prompt)
    return response

def parse_response(response, response_parser):
    parsed_response = response_parser.get_parsed_response(response)
    return parsed_response

def collect_support_data(row, parsed_response):
    support_entry = {
        'comment_id': row['comment_id'],
        'proposition_id': row['proposition_id'],
        'sup_proposition_id': row['sup_proposition_id'],
        'support_boolean': True if 'yes' in parsed_response.lower() else False
    }
    return support_entry

def process_row(index, row, response_parser):
    prompt = create_prompt(row, response_parser)
    response = get_response(prompt)
    parsed_response = parse_response(response, response_parser)
    support_entry = collect_support_data(row, parsed_response)
    return parsed_response, support_entry

def predict_relations(df, response_parser):
    results = []
    support_data = []

    for index, row in df.iterrows():
        parsed_response, support_entry = process_row(index, row, response_parser)
        results.append(parsed_response)
        support_data.append(support_entry)
        
        if index % 10 == 0:
            print(f"Processed {index} proposition pairs.")
    
    print("Prediction of relations completed.")
    df['predicted_relation'] = results
    
    # Optionally convert support_data to a DataFrame for easier processing
    df_support = pd.DataFrame(support_data)

    # Add the support_actual column
    reasons_map = create_reasons_map(data)
    df_support['support_actual'] = df_support.apply(
        lambda row: row['sup_proposition_id'] in reasons_map.get((row['comment_id'], row['proposition_id']), []),
        axis=1
    )

    return df_support

def get_proposition_text(df, comment_id, proposition_id):
    proposition = df[(df['comment_id'] == comment_id) & (df['proposition_id'] == proposition_id)]
    if not proposition.empty:
        return proposition.iloc[0]['text']
    return None

# Compute precision, recall, F1 score, and accuracy
def calculate_metrics(df):
    y_true = df['support_actual']
    y_pred = df['support_boolean']
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
df_support = predict_relations(df, response_parser)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_support)

# Retrieve specific proposition text
text = get_proposition_text(df, comment_id=194, proposition_id=0)
print(text)

# Calculate and print metrics
precision, accuracy, recall, f1 = calculate_metrics(df_support)
print(f'Precision: {precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the result to an Excel file
df_support.to_excel('/home/parklab/data/debugging_dataset/predicted_relations.xlsx', index=False)
