device = "cuda"
cache_dir = "/home/as9df/.cache/huggingface/hub/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir).to(device)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
@dataclass
class Sentence:
    sentence_id: str
    text: str
@dataclass
class Corpus:
    corpus_id: str
    sentences: list
def generate_corpus_structure(corpus_text, corpus_id):
    # sentences = re.split(r'(?<=[.!?;,]) +', corpus_text)
    sentences = sent_tokenize(corpus_text)
    sentence_objects = [Sentence(str(i), sentence.strip()) for i, sentence in enumerate(sentences)]
    return Corpus(str(corpus_id), sentence_objects)
# Mistral setup
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
@dataclass
class SupportResponseParser:
    answer_token: str = "Answer:"
    answer_format: str = "{0}"
    def get_parsed_response(self, response: str) -> str:
        # Split by the answer token and ensure the right part is extracted
        answer = response.strip().split(self.answer_token)[-1].split("\n")[0].strip()
        if answer.lower() in ['yes', 'no']:
            return answer
        return "Invalid response"
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
def determine_support(corpus: Corpus, sim_sentences) -> Dict[str, List[str]]:
    response_parser = SupportResponseParser()
    adjacency_list = {sentence.sentence_id: [] for sentence in corpus.sentences}
    for i, sentence in enumerate(sim_sentences):
    # for i, sentence in enumerate(corpus.sentences):
        # Get the surrounding sentences within the window of 5 indices before or after
        for j in range(max(0, i - 5), min(len(corpus.sentences), i + 6)):
            if i != j:
                surrounding_sentence = corpus.sentences[j]
                prompt = create_prompt(sentence, surrounding_sentence, response_parser)
                # print("\n" + "="*50)
                # print(f"Input to LLM:\n{prompt}")
                response = generate_response(prompt)
                # print("\n" + "-"*50)
                # print(f"LLM Output:\n{response}")
                parsed_response = response_parser.get_parsed_response(response)
                # print("\n" + "-"*50)
                # print(f"Parsed Output:\n{parsed_response}")
                # print("="*50 + "\n")
                if parsed_response.lower() == 'yes':
                    adjacency_list[sentence.sentence_id].append(surrounding_sentence.sentence_id)
                print(i, j)
    return adjacency_list