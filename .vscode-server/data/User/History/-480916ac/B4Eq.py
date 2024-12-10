import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
import re
import time
import nltk

# Download the 'punkt' tokenizer data
nltk.download('punkt')

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

@dataclass
class Sentence:
    sentence_id: str
    text: str

@dataclass
class Corpus:
    corpus_id: str
    sentences: list

def generate_corpus_structure(corpus_text, corpus_id):
    sentences = sent_tokenize(corpus_text)
    sentence_objects = [Sentence(str(i), sentence.strip()) for i, sentence in enumerate(sentences)]
    return Corpus(str(corpus_id), sentence_objects)

@dataclass
class SupportResponseParser:
    answer_token: str = "Answer:"
    answer_format: str = "{0}"

    def get_parsed_response(self, response: str) -> str:
        # Split by the answer token and ensure the right part is extracted
        answer = response.strip().split(self.answer_token)[-1].split("\n")[0].strip()
        return answer

def generate_response(prompt: str, max_length: int = 4096, temperature: float = 0.7, top_p: float = 0.9) -> str:
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

def create_prompt(current_sentence: Sentence, surrounding_sentence: Sentence, response_parser: SupportResponseParser) -> str:
    prompt = (
        "Task Description:\n\n"
        "Objective: Determine whether the second sentence supports the first sentence.\n\n"
        "Instructions:\n"
        "Read the following two sentences carefully:\n"
        "Given the following two sentences:\n"
        f"1. {current_sentence.text}\n"
        f"2. {surrounding_sentence.text}\n"
        "Question: Does the second sentence support the first sentence?\n"
        "With no other words, answer with 'yes' or 'no'.\n"
        "Answer with either 'yes' or 'no', no other words or explanations.\n"
        "Example Answer: Answer: yes\n"
        f"{response_parser.answer_format.format(response_parser.answer_token)}"
    )
    return prompt

def determine_support(corpus: Corpus) -> Dict[str, List[str]]:
    response_parser = SupportResponseParser()
    adjacency_list = {sentence.sentence_id: [] for sentence in corpus.sentences}
    for i, sentence in enumerate(corpus.sentences):
        for j in range(max(0, i - 5), min(len(corpus.sentences), i + 6)):
            if i != j:
                surrounding_sentence = corpus.sentences[j]
                prompt = create_prompt(sentence, surrounding_sentence, response_parser)
                response = generate_response(prompt)
                parsed_response = response_parser.get_parsed_response(response)
                if parsed_response.lower() == 'yes':
                    adjacency_list[sentence.sentence_id].append(surrounding_sentence.sentence_id)
                print(i, j)
    return adjacency_list

def create_reclor_prompt(context, question, choices, support_structure):
    return (
        f"The passage is: {context}\n"
        f"The question is: {question}\n"
        f"Here are 4 choices for it and they are:\n"
        f"{choices}\n"
        f"The following support structure is derived from the passage: {support_structure}\n"
        "Please choose the correct answer based on the support structure provided. Provide only the number (0, 1, 2, or 3) as the final answer."
    )

# Function to isolate the answer from the response
def isolate_answer(response):
    pattern = r"Provide only the number \(0, 1, 2, or 3\) as the final answer\.\s*(.*)"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

# Function to parse the response
def parse_response(response):
    isolated_response = isolate_answer(response)
    print(f"Isolated response:\n{isolated_response}\n")  # Add debug print to see the isolated response
    match = re.search(r'\b([0-3])\b', isolated_response)
    if match:
        return int(match.group(1))
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
    choices = "\n".join([f"{i}. {choice}" for i, choice in enumerate(entry['answers'])])
    true_label = entry['label']
    question_id = entry['id_string']

    # Generate corpus structure from the context
    corpus = generate_corpus_structure(context, question_id)
    support_structure = determine_support(corpus)

    prompt = create_reclor_prompt(context, question, choices, support_structure)
    print(f"Prompt:\n{prompt}\n")
    
    response = generate_response(prompt)
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
