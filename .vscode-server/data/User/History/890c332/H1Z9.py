import json
import os
import regex as re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from typing import List, Optional
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../link-prediction')))
from src.lib.link_prediction.contextual_assistant_response_parser import SupportResponseParser



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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
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

def run_experiment(
    generator,
    response_parser,
    data,
    system_prompt: str,
    user_prompt_format: str,
    temperature: float,
    top_p: float,
    max_batch_size: int,
    max_gen_len: Optional[int],
    run_on_validation=False,
    run_on_test=False
):
    splits_to_use = [data['train']]
    if run_on_validation and data['validation']:
        splits_to_use.append(data['validation'])
    if run_on_test and data['test']:
        splits_to_use.append(data['test'])
        
    support_data = []  # List to keep track of proposition support information
    
    # Create the reasons map to parse actual/true answer from reasons attribute of each proposition
    reasons_map = create_reasons_map([comment for split in splits_to_use for comment in split])
    
    # A list of Comments when use_propositions: false in config, otherwise a list of Propositions in a given dataset that consists of Comments
    for split in splits_to_use:
        user_prompts = []
        for comment in split:
            # Generate proposition pairs for each comment
            pairs = generate_proposition_pairs(comment)
            for prop, surrounding in pairs:
                for s_prop in surrounding:
                    user_prompt = user_prompt_format.format(prop.text, s_prop.text,
                                                  response_parser.answer_format.format(response_parser.answer_token))
                    user_prompts.append(user_prompt)
                    # Track proposition support information
                    support_data.append({
                        'comment_id': comment.id,
                        'proposition_id': prop.id,
                        'sup_proposition_id': s_prop.id,
                        'support_boolean': None
                    })
                    
        examples = [] #TODO: Fix how to pull examples from config
        
        dialogs: List[str] = get_dialogs(system_prompt, user_prompts, examples, True)
        print('------------------------DIALOGS LOADED----------------------')
        
        results = []
        for i in range(0, len(dialogs), max_batch_size):
            dialogs_batch = dialogs[i:i+max_batch_size] if i+max_batch_size < len(dialogs) else dialogs[i:]
            print(f'Processing batch {i//max_batch_size + 1}/{(len(dialogs) + max_batch_size - 1)//max_batch_size}')
            
            print('===========================DIALOGS_BATCH==============================')
            print(f"Dialogs batch: {dialogs_batch}")  # Inspect the batch content
            print('===========================END_DIALOGS_BATCH==============================')
            
            try:
                batch_results = [generate_response(dialog, max_gen_len, temperature, top_p) for dialog in dialogs_batch]
                print(f"Generated results: {batch_results}")
                parsed_results = [response_parser.get_parsed_response(result) for result in batch_results]
                print(f"Parsed results: {parsed_results}")
                results += parsed_results

                print("-------------------------------------------------------------")
                
                # Update support_boolean in support_data based on parsed_results
                for idx, parsed_result in enumerate(parsed_results):
                    support_data[i + idx]['support_boolean'] = True if 'yes' in parsed_result.lower() else False
                    
            except Exception as e:
                print(f"Error during batch processing: {e}")
        
        print("=================================================================")
        
        # Optionally convert support_data to a DataFrame for easier processing
        df_support = pd.DataFrame(support_data)

        # Add the support_actual column
        df_support['support_actual'] = df_support.apply(
            lambda row: row['sup_proposition_id'] in reasons_map.get((row['comment_id'], row['proposition_id']), []),
            axis=1
        )
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df_support)  # Print a sample of the support data
        
        # Extract true and predicted labels
        y_true = df_support['support_actual']
        y_pred = df_support['support_boolean']
        
        # Compute precision, recall, F1 score, and accuracy
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Print results
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

# Example usage:
data = read_jsonl(filename)
dataset = create_dataset(data)

# Assuming `generate_proposition_pairs` and `create_reasons_map` are available
response_parser = SupportResponseParser()
system_prompt = "System prompt here"  # Define your system prompt
user_prompt_format = "Classify the relation between the following propositions: {} and {}. {}"

# Example of how to call run_experiment
run_experiment(
    generator=model,
    response_parser=response_parser,
    data=dataset,
    system_prompt=system_prompt,
    user_prompt_format=user_prompt_format,
    temperature=0.7,
    top_p=0.9,
    max_batch_size=8,
    max_gen_len=256,
    run_on_validation=False,
    run_on_test=False
)
