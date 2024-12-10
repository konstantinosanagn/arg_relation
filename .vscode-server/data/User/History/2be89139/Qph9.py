import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple

# Load the dataset from a given path
DATASET_PATH = 'reclor_sample.json' 

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data

# Load the dataset
data = load_dataset(DATASET_PATH)

# Pretty-print the loaded data
print("Loaded Data:")
print(json.dumps(data, indent=4))

# Load Mistral v0.3 model and tokenizer from Hugging Face
print("Loading Mistral v0.3 model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Mistral v0.3 model loaded successfully and moved to device.")

# Function to generate a response using the model
def generate_response(model, tokenizer, prompt: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to create a prompt
def create_prompt(context, question, choices):
    return (
        f"The passage is: {context}\n"
        f"The question is: {question}\n"
        f"Here are 4 choices for it and they are:\n"
        f"{choices}\n"
        "Which one should I choose?"
    )

# Function to parse the response
def parse_response(response):
    choices = ['A', 'B', 'C', 'D']
    for choice in choices:
        if choice.lower() in response.lower():
            return choices.index(choice)
    return -1

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Iterate over the data
for entry in data:
    context = entry['context']
    question = entry['question']
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(entry['answers'])])
    true_label = entry['label']
    
    prompt = create_prompt(context, question, choices)
    print(f"Prompt:\n{prompt}\n")
    
    response = generate_response(model, tokenizer, prompt)
    print(f"Generated response:\n{response}\n")
    
    predicted_label = parse_response(response)
    print(f"Predicted label: {predicted_label}\n")
    
    y_true.append(true_label)
    y_pred.append(predicted_label)

# Calculate and print metrics
precision = precision_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
