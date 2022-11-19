from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

# https://qiita.com/ground0state/items/155b77f4c07e1a509a14

input_str = input(str("入力する文字列: "))

count = CountVectorizer()
docs = np.array(['The sun is shining', 
                 'The weather is shining',
                 'The sun is shining, the weather is sweet, and one and one it two.'])
bag = count.fit_transform(docs)
count_dict = count.vocabulary_
count_dict = sorted(count_dict.items())
count_dict = dict((x, y) for x, y in count_dict)
count_keys:list = list(count_dict.keys())

# 入力した文章の分かち書きを出力。アルファベット順にしてある。
print(count_keys)

# 文章の数だけリストがある。その文章での単語の出現回数を出力。順番は、アルファベット順で先ほどのcount_keysと一緒。
print(bag.toarray())

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
# 上記の文章に対してのtf-idf値を出力
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())