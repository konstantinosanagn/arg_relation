import json
import os
import regex as re
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

# Load Gemma model and tokenizer from Hugging Face
print("Loading Gemma model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Gemma model loaded successfully and moved to device.")

def generate_response(prompt: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
    print(f"Input prompt: {prompt}")  # Print the input prompt
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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Model response: {response}")  # Print the model's response
    return response

class PropositionResponseParser:
    """
    Parses the proposition classification from an assistant response.
    """
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='Classification: {}'):
        """
        See parent.
        """
        self.answer_token = answer_token
        self.answer_format = answer_format

    def _get_answer_using_regex_match(self, pattern: str, response: str) -> str:
        match: re.Match = re.search(pattern, response, re.IGNORECASE)
        result = match.group(1) if match else ""
        return result

    def get_parsed_response(self, response: str) -> str:
        """
        See parent.
        """
        proposition_types = set(['fact', 'testimony', 'policy', 'value', 'reference'])
        patterns = [
            r'\b(fact|testimony|policy|value|reference)\b',
            r'.*?Classification:\s*(\w+)',
            r'.*?"(\w+)"'
        ]
        print(f"Parsing assistant response: {response}")
        for pattern in patterns:
            search_res = self._get_answer_using_regex_match(pattern, response)
            if search_res:
                break
        else:
            return "unknown"
        possible_ans = search_res.strip().lower()
        print(f"Possible parsed response: {possible_ans}")
        return possible_ans if possible_ans in proposition_types else "unknown"

def predict_types(df, response_parser):
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
        parsed_response = response_parser.get_parsed_response(response)
        df.at[index, 'predicted_type'] = parsed_response

        print("------------------------------------------------------------------------------------------")
        print(f"Input: {row['text']}")  # Print the input text
        print(f"Response: {response}")  # Print the response from the model
        print(f"Parsed Response: {parsed_response}")  # Print the parsed response
        print("------------------------------------------------------------------------------------------")
        
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

# Initialize the response parser
response_parser = PropositionResponseParser()

# Predict types
df = predict_types(df, response_parser)
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
