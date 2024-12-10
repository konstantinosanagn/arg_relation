import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from huggingface_hub import login

# Replace this with your actual Hugging Face token
access_token = "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"

# Log in to Hugging Face
login(token=access_token)

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
            actual_type = prop['type']
            dataset.append({
                'comment_id': comment_id,
                'proposition_id': proposition_id,
                'text': text,
                'actual_type': actual_type,
                'predicted_type': None  # This will be filled later
            })
    print("Dataset created successfully.")
    return pd.DataFrame(dataset)

# Load Mistral v0.2 model and tokenizer from Hugging Face
print("Loading Mistral v0.2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Mistral v0.2 model loaded successfully and moved to device.")

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

def predict_types(df):
    for index, row in df.iterrows():
        prompt = (
            "The five types of propositions are 'fact', 'testimony', 'policy', 'value', and 'reference'. "
            "'Fact' is an objective proposition, meaning there are no subjective interpretations or judgments. "
            "'Testimony' is also an objective proposition that is experiential. "
            "'Policy' is a subjective proposition that insists on a specific course of action. "
            "'Value' is a subjective proposition that is a personal opinion or expression of feeling. "
            "'Reference' refers to a resource containing objective evidence. In product reviews, reference is usually a URL to another product page, image, or video.\n\n"
            "Classify the following proposition as 'fact', 'testimony', 'policy', 'value', or 'reference': "
            f"{row['text']}"
            " With no other text, answer in the format:\nClassification: <answer>"
        )
        response = generate_response(prompt)
        response_lower = response.lower()
        
        if "fact" in response_lower:
            predicted_type = "fact"
        elif "testimony" in response_lower:
            predicted_type = "testimony"
        elif "policy" in response_lower:
            predicted_type = "policy"
        elif "value" in response_lower:
            predicted_type = "value"
        elif "reference" in response_lower:
            predicted_type = "reference"
        else:
            predicted_type = 'unknown'
            
        df.at[index, 'predicted_type'] = predicted_type
        
        if index % 10 == 0:
            print(f"Processed {index} propositions.")
    print("Prediction of types completed.")
    return df

def get_proposition_text(df, comment_id, proposition_id):
    proposition = df[(df['comment_id'] == comment_id) & (df['proposition_id'] == proposition_id)]
    if not proposition.empty:
        return proposition.iloc[0]['text']
    return None

# Compute precision, recall, F1 score, and accuracy
def calculate_metrics(df):
    y_true = df['actual_type']
    y_pred = df['predicted_type']
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    return precision, accuracy, recall, f1

# Read data from uploaded file
data = read_jsonl(filename)

# Create dataset
df = create_dataset(data)

# Predict types
df = predict_types(df)
print(df.head())

# Retrieve specific proposition text
text = get_proposition_text(df, comment_id=194, proposition_id=0)
print(text)

# Calculate and print metrics
precision, accuracy, recall, f1 = calculate_metrics(df)
print(f'Precision: {precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
