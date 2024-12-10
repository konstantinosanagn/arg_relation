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
    sentence_objects = [Sentence(str(i + 1), sentence.strip()) for i, sentence in enumerate(sentences)]
    return Corpus(str(corpus_id), sentence_objects)


