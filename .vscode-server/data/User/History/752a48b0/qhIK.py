import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple

# Define the classes and methods

class SupportResponseParser:
    def __init__(self, answer_token: str = '<answer>', answer_format: str = 'Answer: {}'):
        self.answer_token = answer_token
        self.answer_format = answer_format

    def _get_answer_using_regex_match(self, pattern: str, response: str) -> str:
        match = re.search(pattern, response, re.IGNORECASE)
        return match.group(1) if match else ""

    def get_parsed_response(self, response: str) -> str:
        patterns = [
            r'<answer>\s*(yes|no)\s*</answer>',
            r'".*?\b(yes|no)\b.*?"',
            r'\b(yes|no)\b',
            r'\b(Yes|No)\b',
            r'\b(support|attack)\b',
            r'".*?\b(support|attack)\b.*?"'
        ]
        for pattern in patterns:
            search_res = self._get_answer_using_regex_match(pattern, response)
            if search_res:
                return search_res.strip().lower()
        return "unknown"

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

def create_prompt(row, response_parser):
    return (
        "Task Description:\n\n"
        "Objective: Determine whether the second sentence supports the first sentence.\n\n"
        "Instructions:\n"
        "Read the following two sentences carefully:\n"
        "Given the following two sentences:\n"
        f"1. {row['text']}\n"
        f"2. {row['sup_text']}\n"
        f"Question: Does the second sentence support the first sentence?\nWith no other words, answer with 'yes' or 'no'.\n"
        f"{response_parser.answer_format.format(response_parser.answer_token)}"
    )

def get_response(model, tokenizer, prompt):
    return generate_response(model, tokenizer, prompt)

def parse_response(response, response_parser):
    return response_parser.get_parsed_response(response)

def collect_support_data(row, parsed_response):
    return {
        'comment_id': row['comment_id'],
        'proposition_id': row['proposition_id'],
        'sup_proposition_id': row['sup_proposition_id'],
        'support_boolean': True if 'yes' in parsed_response.lower() else False
    }

def process_row(index, row, model, tokenizer, response_parser):
    prompt = create_prompt(row, response_parser)
    response = get_response(model, tokenizer, prompt)
    parsed_response = parse_response(response, response_parser)
    support_entry = collect_support_data(row, parsed_response)
    return parsed_response, support_entry

def predict_relations(df, model, tokenizer, response_parser):
    results = []
    support_data = []

    for index, row in df.iterrows():
        parsed_response, support_entry = process_row(index, row, model, tokenizer, response_parser)
        results.append(parsed_response)
        support_data.append(support_entry)
        
        if index % 10 == 0:
            print(f"Processed {index} proposition pairs.")
    
    print("Prediction of relations completed.")
    df['predicted_relation'] = results
    
    df_support = pd.DataFrame(support_data)

    reasons_map = create_reasons_map(df.to_dict('records'))
    df_support['support_actual'] = df_support.apply(
        lambda row: row['sup_proposition_id'] in reasons_map.get((row['comment_id'], row['proposition_id']), []),
        axis=1
    )

    return df_support

def calculate_metrics(df):
    y_true = df['support_actual']
    y_pred = df['support_boolean']
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, accuracy, recall, f1

def run_experiment(config_path, model, tokenizer):
    with open(config_path, 'r') as file:
        cfg = json.load(file)

    data = read_jsonl(cfg['train_path'])
    df = create_dataset(data)

    response_parser = SupportResponseParser()

    df_support = predict_relations(df, model, tokenizer, response_parser)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_support)

    precision, accuracy, recall, f1 = calculate_metrics(df_support)
    print(f'Precision: {precision:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    df_support.to_excel('/home/parklab/data/debugging_dataset/predicted_relations.xlsx', index=False)

def main(config_path: str):
    print("Loading Mistral v0.3 model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Mistral v0.3 model loaded successfully and moved to device.")

    run_experiment(config_path, model, tokenizer)

if __name__ == '__main__':
    import fire
    fire.Fire(main)