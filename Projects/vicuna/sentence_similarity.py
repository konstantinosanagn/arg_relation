import nltk
from nltk.tokenize import sent_tokenize
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_sentence_embedding(sentence):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings
def calculate_similarity(s1, s2):
    return 1 - cosine(get_sentence_embedding(s1).numpy(), get_sentence_embedding(s2).numpy())
def get_sorted_sentences(context, question):
    sentences = sent_tokenize(context)
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = calculate_similarity(question, sentence)
        sentence_scores[sentence] = [score, i]
    sorted_sentences = sorted(sentence_scores.items(), key=lambda item: item[1][0], reverse=True)
    return sorted_sentences