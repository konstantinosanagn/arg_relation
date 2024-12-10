import re
from dataclasses import dataclass

@dataclass
class Sentence:
    sentence_id: str
    text: str

@dataclass
class Corpus:
    corpus_id: str
    sentences: list

def generate_corpus_structure(corpus_text, corpus_id):
    sentences = re.split(r'(?<=[.!?;,]) +', corpus_text)
    sentence_objects = [Sentence(str(i), sentence.strip()) for i, sentence in enumerate(sentences)]
    return Corpus(str(corpus_id), sentence_objects)


import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Assuming `Sentence` and `Corpus` classes and `generate_corpus_structure` function are already defined.

# Mistral setup
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@dataclass
class SupportResponseParser:
    answer_token: str = "Answer:"
    answer_format: str = " {0} yes or no."

    def get_parsed_response(self, response: str) -> str:
        return response.strip().split(self.answer_token)[-1].strip()

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

def create_prompt(current_sentence: Sentence, surrounding_sentence: Sentence, response_parser: SupportResponseParser) -> str:
    prompt = (
        "Task Description:\n\n"
        "Objective: Determine whether the second sentence supports the first sentence.\n\n"
        "Instructions:\n"
        "Read the following two sentences carefully:\n"
        "Given the following two sentences:\n"
        f"1. {current_sentence.text}\n"
        f"2. {surrounding_sentence.text}\n"
        f"Question: Does the second sentence support the first sentence?\nWith no other words, answer with 'yes' or 'no'.\n"
        f"{response_parser.answer_format.format(response_parser.answer_token)}"
    )
    return prompt

def determine_support(corpus: Corpus) -> Dict[str, List[str]]:
    response_parser = SupportResponseParser()
    adjacency_list = {sentence.sentence_id: [] for sentence in corpus.sentences}

    for i, sentence in enumerate(corpus.sentences):
        # Get the surrounding sentences within the window of 5 indices before or after
        for j in range(max(0, i - 5), min(len(corpus.sentences), i + 6)):
            if i != j:
                surrounding_sentence = corpus.sentences[j]
                prompt = create_prompt(sentence, surrounding_sentence, response_parser)
                response = generate_response(prompt)
                parsed_response = response_parser.get_parsed_response(response)
                if 'yes' in parsed_response.lower():
                    adjacency_list[sentence.sentence_id].append(surrounding_sentence.sentence_id)
    
    return adjacency_list

# Example usage
corpus_text = "Hello there! How are you doing today? I hope you're having a great day; let's catch up soon."
corpus_id = 0
corpus = generate_corpus_structure(corpus_text, corpus_id)
adjacency_list = determine_support(corpus)

print(adjacency_list)
