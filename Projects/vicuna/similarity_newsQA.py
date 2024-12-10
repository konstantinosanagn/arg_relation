from sentence_similarity import get_sorted_sentences
from relation_detection_for_similarity import generate_corpus_structure, determine_support
from data_generators import gen_newsQA, gen_nq, gen_reclor
import time
import nltk
from nltk.tokenize import sent_tokenize
a = time.time()
data = gen_reclor()
passage = data['context']
question = data['question']
answer = data['answer']
link_prediction_structure = generate_corpus_structure(passage, 0)
adjacency_list = determine_support(link_prediction_structure)
sim_sentences = get_sorted_sentences(passage, question)
sentences = sent_tokenize(passage)
for i, sentence in enumerate(sentences):
    print(f"Sentence {i}: {sentence}")
print(f"\nAnswer: {answer}\n")
print(f"Adjacency list: {adjacency_list}\n")
print(f"Top similar sentences: {sim_sentences}\n")
b = time.time()
print(f"Time elapsed: {b - a}")