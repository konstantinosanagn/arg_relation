import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import time

# Load the dataset from a given path
DATASET_PATH = 'reclor_sample.json'  # replace with the actual path to your dataset

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
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
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

# Function to isolate the answer from the response
def isolate_answer(response):
    parts = response.split('[/INST]')
    last_part = parts[-1] if len(parts) > 1 else parts[0]
    cleaned_response = last_part.strip()
    return cleaned_response

# Function to parse the response
def parse_response(response):
    isolated_response = isolate_answer(response)
    isolated_response = isolated_response.lower()  # convert response to lowercase for easier matching
    isolated_response = re.sub(r'[^a-d]', '', isolated_response)  # remove all non a-d characters
    if 'a' in isolated_response:
        return 0
    elif 'b' in isolated_response:
        return 1
    elif 'c' in isolated_response:
        return 2
    elif 'd' in isolated_response:
        return 3
    else:
        return -1

# Initialize lists to store true and predicted labels and question IDs
y_true = []
y_pred = []
question_ids = []

# Start the timer and set the counter
start_time = time.time()
question_count = 0

# Iterate over the data
for entry in data:
    context = entry['context']
    question = entry['question']
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(entry['answers'])])
    true_label = entry['label']
    question_id = entry['id_string']
    
    prompt = create_prompt(context, question, choices)
    print(f"Prompt:\n{prompt}\n")
    
    response = generate_response(model, tokenizer, prompt)
    print(f"Generated response:\n{response}\n")
    
    predicted_label = parse_response(response)
    print(f"Predicted label: {predicted_label}\n")
    
    y_true.append(true_label)
    y_pred.append(predicted_label)
    question_ids.append(question_id)
    
    question_count += 1

# Stop the timer
end_time = time.time()
total_time = end_time - start_time
average_time_per_question = total_time / question_count

# Calculate and print metrics
precision = precision_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Total time: {total_time:.2f} seconds')
print(f'Average time per question: {average_time_per_question:.2f} seconds')

# Create and print DataFrame with question IDs, true labels, and predicted labels
df_results = pd.DataFrame({
    'Question ID': question_ids,
    'True Label': y_true,
    'Predicted Label': y_pred
})

print("\nResults DataFrame:")
print(df_results)