import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

x = np.array([0.789, 0.515, 0.335, 0])
y = np.array([0.832, 0.555, 0, 0])
print(f'cos類似度: {cos_sim(x, y)}')

count = CountVectorizer()
docs = np.array(['The sun is shining', 
                 'The weather is shining',
                 'The sun is shining, the weather is sweet, and one and one it two.'])
bag = count.fit_transform(docs)
print(count.vocabulary_)

bag = count.fit_transform(docs)
print(bag.toarray())


tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())