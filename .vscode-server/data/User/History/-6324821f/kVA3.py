import pandas as pd
import random as r

def rand_validation_news():
    i = r.randint(0, 99)
    val = [20213, 46975, 8303, 81686, 118129, 21186, 73486, 102243, 43840, 116746, 37841, 106510, 69823, 49147, 48880, 10305, 116309, 54780, 104536, 49920, 2577, 111159, 109858, 116478, 6846, 21926, 32057, 641, 59778, 12385, 19770, 101669, 86068, 110203, 9697, 108294, 114915, 73749, 24152, 52561, 13885, 37199, 28579, 73433, 1658, 65543, 4633, 32222, 96211, 61399, 22293, 89533, 89606, 14542, 71318, 77481, 119040, 21064, 82004, 116093, 9729, 81964, 47073, 34951, 28858, 91784, 106361, 63, 36459, 76511, 54642, 88912, 49020, 117121, 12026, 1661, 50125, 36594, 20414, 119389, 85261, 11529, 118707, 32182, 68055, 86685, 18769, 72011, 104975, 75086, 13418, 95629, 20233, 110848, 7958, 22389, 37242, 94645, 39900, 47713]
    return val[i]

def gen_newsQA_helper():
    i = rand_validation_news() # 119632
    df = pd.read_csv("/home/parklab/data/newsqa-data-v1.csv")
    story_id = "/home/parklab/data/" + str(df.iloc[i]['story_id']).replace("./cnn", "cnn/cnn")
    with open(story_id, 'r') as file:
        story = file.read()
    question = df.iloc[i]['question']
    ans_ranges = df.iloc[i]['answer_char_ranges'].split("|")
    answers = []
    for a in ans_ranges:
        if a == "None":
            continue
        tmp = a.split(",")
        for b in tmp:
            ans_range = b.split(":")
            answers.append(story[int(ans_range[0]):int(ans_range[1])])
    if df.iloc[i]['is_question_bad'] == "?" or df.iloc[i]['is_answer_absent'] == "?":
        return None
    if float(df.iloc[i]['is_question_bad']) < .5 and float(df.iloc[i]['is_answer_absent']) < .5:
        return {'context': story, 'question': question, 'answers': answers}
    else:
        return None

def gen_newsQA():
    while True:
        tmp = gen_newsQA_helper()
        if tmp != None:
            return tmp

# Collect 30 sample passages
samples = []
for _ in range(30):
    samples.append(gen_newsQA())

# Initialize a list to store the data
data = []

# Print each passage and its MC options, and get user input for the correct answer
for idx, sample in enumerate(samples):
    print(f"Sample {idx+1}:\n")
    print("Passage:\n", sample['context'])
    print("\nQuestion:\n", sample['question'])
    print("\nAnswers:")
    for i, answer in enumerate(sample['answers']):
        print(f"{i + 1}. {answer}")
    print("\n")

    # Get user input for the correct answer
    correct_answer_input = input("Which sentence or sentences contain the correct answer? (Provide indices separated by commas if multiple): ")
    
    # Add the data to the list
    data.append({
        'context': sample['context'],
        'question': sample['question'],
        'answers': sample['answers'],
        'correct_answer': sample['answers'][0] if sample['answers'] else None,  # Assume the first answer is the correct one
        'user_input': correct_answer_input
    })
    print("\n" + "="*80 + "\n")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
df.to_csv('/home/parklab/data/newsqa_sampled.csv', index=False)

print("Sampled dataset saved to /home/parklab/data/newsqa_sampled.csv")
