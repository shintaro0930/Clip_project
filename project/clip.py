import os
import re
import sys
sys.path.append('CLIP')
import torch
from CLIP import clip
from PIL import Image
from googletrans import Translator
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
import pyheif
import glob
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def remove_punctuation(input):
  output = re.sub(r'[^\w\s]','',input)
  return output

# textを分かち書きの形にする
def wakachi(text)->list:
  t = Tokenizer()
  tokens = t.tokenize(text)
  docs = []
  for token in tokens:
    docs.append(token.surface)
  return docs

def vecs_array(documents):
  docs = np.array(documents)
  vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
  vecs = vectorizer.fit_transform(docs)
  return vecs.toarray()


  doc = np.array(document)
  doc = [doc]
  vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
  vecs = vectorizer.fit_transform(doc)
  return vecs.toarray()

# heicファイルをpngに変換
def heic_png(image_path, save_path):
    heif_file = pyheif.read(image_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    # JPEGで保存
    images.append(data.save(str(save_path), "JPEG"))

text_base_dir = '/work/texts/'
# image_base_dir = '/work/pictures/'
image_base_dir = '/work/light_pictures/'


texts_jp:list = []
texts_dir = os.listdir(text_base_dir)

text_file = text_base_dir + 'text.txt'
# text_file = text_base_dir + 'rand_text.txt'

with open(text_file) as texts:
  for text in texts:
    text = re.sub(r'[^\w\s]', '', text)
    texts_jp.append(text.rstrip())    #.rstrip()は改行コードを消す

input_text = input("input: ")
input_text = re.sub(r'[^\w\s]', '', input_text)

texts_jp.append(input_text)

# 入力文のベクトル化
input_vector = vecs_array(texts_jp)[-1]



cos_sim_array = np.round(cosine_similarity(vecs_array(texts_jp), vecs_array(texts_jp)), len(texts_jp))
input_cos_list:list = cos_sim_array[-1]
cos_sim_dict:dict = dict(zip(input_cos_list, texts_jp))
cos_sim_dict.pop(1)
cos_sim_dict = sorted(cos_sim_dict.items(), reverse=True)
cos_sim_dict = dict((x, y) for x, y in cos_sim_dict)

texts_jp.pop(-1)
images = []
files = os.listdir(image_base_dir)

# .heic, .HEICを消し去りたい
for file in files:
  full_path_file = image_base_dir + file
  root_extenstion_tuple = os.path.splitext(file) # root_extension_tuple: tuple型
  if(root_extenstion_tuple[1] == '.heic' or root_extenstion_tuple[1] == '.HEIC'):
    before_image = str(full_path_file)
    after_image = image_base_dir + root_extenstion_tuple[0] + '.jpg'
    heic_png(before_image, after_image)
    os.remove(file)     #.heic or .HEICを削除
    continue
  elif(root_extenstion_tuple[1] == '.sh'):
    continue
  images.append(file)

translator = Translator()
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]

clip_dict:dict = {}
clip_text:list = []
clip_cos_list = []
for i, image in enumerate(images):
  try:
    original_image = Image.open(image_base_dir + f"{image}")
    print(f'--- {images[i]} ---')
    image = preprocess(original_image).unsqueeze(0).to(device)
    text = clip.tokenize(texts_en).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)   # model.pyの345行目
        logits_per_image, logits_per_text = model(image, text)
        #logits_per_image: 画像に対するテキストの一致度
        #logits_per_text: テキストに対する画像の一致度
        # コーパスの確率を保存
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs2 = logits_per_text.softmax(dim=-1).cpu().numpy().reshape([1, -1])
        sorted_probs = np.sort(probs)
        index = np.argsort(probs)

    # 少なくともここで、{画像、ソートした上位3種:その確率}を保持

    # print("=== NORMAL===")
    # for i in range(probs.shape[-1]):
    #     text = texts_jp[i] + "(" + texts_en[i] + ")"
    #     print(f'type(text): {type(text)}')
    #     print(f'type(probs): {type(probs[0, i])}')
    #     print(f'{text}: {probs[0, i]*100:0.1f}%')
    #     # clip_dict.update(text=probs[0, i]*100)
    #     # print(clip_dict)
    #     # print(f'{texts_jp[i]}({texts_en[i]}): {probs[0, i]*100:0.1f}%')
    
    print("===REVERSE===")    # ソートして上位3つ出力
    for i in reversed(range(sorted_probs.shape[-1] - 3, sorted_probs.shape[-1])):
        clip_text.append(texts_jp[index[0, i]])
        print(f'{texts_jp[index[0, i]]}({texts_en[index[0, i]]}): {sorted_probs[0, i]*100:0.1f}%')
    print("\n")

    clip_text.append(input_text)
    clip_cos_sim = np.round(cosine_similarity(vecs_array(clip_text), vecs_array(clip_text)), len(clip_text))
    clip_cos_list = clip_cos_sim[-1]
    print("=============")
    print(f'入力文とソート文のcos類似度:\n {clip_cos_list}')
    print("=============")    


    # # cos_simを出す
    # # input_vectorとclip_text
    # input_vector = np.array(input_vector).reshape((-1, 1))
    # input_vector = np.reshape(input_vector, (12, 1))
    # print(input_vector.shape)
    # print("==========")
    # print(vecs_array(clip_text).shape)
    # cos_sim = np.round(cosine_similarity(input_vector, vecs_array(clip_text)))
    # print("=============")
    # print(f'cos_sim:\n {cos_sim}')
    # print("=============")

  except Exception as e:
    print(e)
    continue



