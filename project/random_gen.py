import random
import os
from googletrans import Translator

# https://wak-tech.com/archives/1625
translator = Translator()

# import adjectives list
with open("EnglishList/english-adjectives.txt") as f:
    adjectives = f.readlines()
    
# import nouns list
with open("EnglishList/english-nouns.txt") as f:
    nouns = f.readlines()

with open('./texts/rand_text.txt', 'a') as f:
    for i in range(100):
        adj_index = random.randrange(len(adjectives))
        adj_index2 = random.randrange(len(adjectives))
        noun_index = random.randrange(len(nouns))
        theme = adjectives[adj_index2] + ' '+adjectives[adj_index]+' '+nouns[noun_index]
        theme = theme.replace("\n", '')
        print(theme)
        f.writelines(translator.translate(theme, dest="ja", src="en").text + '\n')