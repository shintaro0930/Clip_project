import numpy as np
import re
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#わかち書き関数
def wakachi(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    # print(docs)
    return docs
 
#文書ベクトル化関数

def remove_punctuation(input):
  output = re.sub(r'[^\w\s]','',input)
  return output

def vecs_array(documents):
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()
 
if __name__ == '__main__':
    input_text = input("入力文字列: ")
    docs = [
    "私は犬が好きです。",
    "私は犬が嫌いです。",
    "私は犬のことがとても好きです。"]
    docs.append(input_text)
    print(docs)
 
    for doc in docs:
        doc = remove_punctuation(doc)
        print(f'punctuation: {doc}')
    
    #類似度行列作成
    cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)

    # それぞれのtf-idfを取得
    for i, doc in enumerate(docs):
        max_tf_idf = 0
        print(f'{doc}に対して:')
        for j, str in enumerate(docs):
            if(i == j):
                continue
            else:
                if(cs_array[i][j] > max_tf_idf):
                    max_tf_idf = cs_array[i][j]
                    print(f'{str}: {cs_array[i][j]}')
        if(max_tf_idf == 0):
            print(f'{str}はtf-idfが0')
        print()




