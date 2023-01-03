import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

text_base_dir = '/work/texts/'

# ReLU function
def relu(x:int)->int:
    return x * (x > 0.0)

# remove '\r' or '\r\n' and punctuations
def remove_punctuation(input):
    output = re.sub(r'[^\w\s]','',input)
    output = output.rstrip()
    return output

# change text to the style of wakachi
def wakachi(text)->list:
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs:list = []
    for token in tokens:
        docs.append(token.surface)
    return docs

def vecs_array(documents):
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()

# ========================================
input_text = input("input:")
# input_text = "夜景"
input_text = remove_punctuation(input_text)

with open('output.txt') as f:
    lines:list = f.readlines()

line_list = []
prob_list = []
line_list_list = []
prob_list_list = []
count = 0
for line in lines:
    if(re.search('[=]+', line)):
        continue
    count += 1
    line = line.rstrip()
    prob = line
    line = line.split(':')[0]
    line_list.append(line)
    prob = prob.split(' ')[-1]
    prob_list.append(prob)
    if(count % 4 == 0):
        line_list_list.append(line_list)
        prob_list_list.append(prob_list)
        line_list = []
        prob_list = []


max_prob = 0
save_images = []        # 最終的にoutputする画像を保持
#print(line_list_list)
for (line_list,prob_list) in zip(line_list_list, prob_list_list):
    image_name = line_list.pop(0)
    line_list.append(input_text)    
    cos_sim = np.round(cosine_similarity(vecs_array(line_list), vecs_array(line_list)), len(line_list))
    cos_list = cos_sim[-1].tolist()           # .tolist()で numpy.ndarray --> list
    cos_list.pop(-1)
    output_avg = sum(cos_list)/ len(cos_list) * 100      # %表示
    output_avg = np.round(output_avg, decimals=2)
    print(f'{image_name}と{input_text}の類似度: {output_avg}%')
    if(output_avg > max_prob):
        save_images.clear()
        save_images.append(image_name)
        max_prob = output_avg
    elif (output_avg == max_prob):
        save_images.append(image_name)

if(max_prob == 0):
    print('入力文に合う画像はありません')
else:
    with open(f'{text_base_dir}prob_output.txt', 'a', encoding='utf-8', newline='\n') as f:
        f.write('=============\n')
        f.write(f'入力文に合う確率は{max_prob:0.2f}%です。その画像は\n')
        print(f'入力文に合う確率は{max_prob:0.2f}%です。その画像は\n')
        for image in save_images:
            print(image)
            f.write(f'{image}\n')