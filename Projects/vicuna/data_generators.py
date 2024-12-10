import random as r
import pandas as pd
import json
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
            answers.append(repr(story)[int(ans_range[0])-10:int(ans_range[1])+10])
    if df.iloc[i]['is_question_bad'] == "?" or df.iloc[i]['is_answer_absent'] == "?":
        return None
    if float(df.iloc[i]['is_question_bad']) < .5 and float(df.iloc[i]['is_answer_absent']) < .5:
        return {'context':story, 'question':question, 'answers':answers}
    else:
        return None
def gen_newsQA():
    while True:
        tmp = gen_newsQA_helper()
        if tmp != None:
            return tmp
def rand_validation_nq():
    i = r.randint(0, 99)
    val = [1541, 536, 443, 776, 988, 208, 1404, 397, 1511, 28, 1253, 1271, 138, 1571, 716, 318, 1136, 1552, 392, 514, 1203, 1269, 1024, 819, 1055, 857, 220, 246, 1614, 1, 1458, 1005, 1150, 1412, 544, 957, 319, 1675, 175, 1531, 193, 1399, 797, 1572, 1617, 1724, 876, 1421, 9, 1374, 1290, 437, 1759, 312, 1610, 247, 1511, 1168, 292, 846, 204, 255, 763, 145, 756, 265, 1482, 938, 209, 129, 1247, 521, 1672, 334, 500, 330, 748, 229, 1165, 895, 1014, 1226, 1630, 1490, 727, 1505, 1221, 86, 1663, 80, 1354, 1104, 1262, 1144, 1131, 523, 531, 1019, 709, 1445]
    return val[i]
def gen_nq():
    i = rand_validation_nq()
    df = pd.DataFrame(json.load(open('/home/parklab/data/nqdata.json', 'r'))).iloc[i]['data']
    context = df['context']
    long_answer = df['long_answer']
    question = df['question']
    answer = df['short_answer']
    return {"context": context, "question": question, "answer": answer, "long_answer": long_answer}
def rand_validation_reclor():
    i = r.randint(0, 99)
    val = [169, 168, 482, 127, 468, 250, 190, 280, 270, 472, 283, 315, 265, 11, 234, 133, 399, 28, 475, 258, 192, 245, 226, 439, 185, 290, 218, 59, 125, 369, 151, 278, 342, 77, 414, 8, 367, 85, 418, 217, 477, 285, 148, 320, 107, 238, 49, 363, 277, 248, 150, 462, 122, 298, 70, 492, 66, 417, 423, 215, 465, 55, 130, 82, 84, 377, 362, 208, 426, 61, 297, 15, 452, 404, 5, 273, 256, 137, 172, 412, 288, 231, 303, 100, 183, 230, 287, 67, 113, 45, 427, 10, 296, 219, 453, 153, 179, 357, 42, 373]
    return val[i]
def gen_reclor():
    # Generates a random context, question, choices, answer quadruplet from the full ReClor dataset
    i = rand_validation_reclor() # 500
    df = pd.DataFrame(json.load(open('/home/parklab/data/reclor.json', 'r'))).iloc[i]
    context = df['context']
    question = df['question']
    choices = ""
    for i, choice in enumerate(df['answers']):
        choices += (str(i + 1) + ") " + choice + "\n")
    answer = str(df['label'] + 1) + ") " + str(df['answers'][df['label']])
    return {"context": context, "question": question, "choices": choices, "answer": answer}