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

# choose the device 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# initialize base directory
text_base_dir = '/work/texts/'
image_base_dir = '/work/pictures/'
# image_base_dir = '/work/light_pictures/'

# change .heic or .HEIC to .jpg and remove these files
def change_extension(files):
  for file in files:
    image_base_dir = '/work/pictures/'
    full_path_file = image_base_dir + file
    root_extenstion_tuple = os.path.splitext(file)
    if(root_extenstion_tuple[1] == '.heic' or root_extenstion_tuple[1] == '.HEIC'): 
      before_image = str(full_path_file)
      after_image = image_base_dir + root_extenstion_tuple[0] + '.jpg'
      heic_png(before_image, after_image)
      try: 
        os.remove(file)
      except Exception as e:
        continue
      images.append(file)
  
  return images

# remove '\r' or '\r\n' and punctuations
def remove_punctuation(input):
  output = re.sub(r'[^\w\s]','',input)
  output = output.rstrip()
  return output

# change text to the style of wakachi
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

#  change extension marks
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

# initialize text lists and designate the full path
texts_jp:list = []
texts_dir = os.listdir(text_base_dir)
text_file = text_base_dir + 'text.txt'
# text_file = text_base_dir + 'rand_text.txt'

with open(text_file) as texts:
  for text in texts:
    texts_jp.append(remove_punctuation(text))

input_text = input("input: ")
input_text = remove_punctuation(input_text)
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
    try:
      os.remove(file)     #.heic or .HEICを削除
    except Exception as e:
      continue
  elif(root_extenstion_tuple[1] == '.sh'):
    continue
  images.append(file)

translator = Translator()
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]


# initialize
clip_text:list = []
clip_cos_list = []
max_prob = 0

for i, image in enumerate(images):
  try:
    save_image = image
    original_image = Image.open(image_base_dir + f"{image}")
    image = preprocess(original_image).unsqueeze(0).to(device)
    text = clip.tokenize(texts_en).to(device)
    clip_text.clear()

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)        # model.pyの345行目
        logits_per_image, logits_per_text = model(image, text)      # imageとtextは転置の関係

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        sorted_probs = np.sort(probs)
        index = np.argsort(probs)
    
    # print("===PROBABILITIES===") 
    # ソートした上位3つのテキスト文章をlistに追加 
    for i in reversed(range(sorted_probs.shape[-1] - 3, sorted_probs.shape[-1])):
        clip_text.append(texts_jp[index[0, i]])
        # print(f'{texts_jp[index[0, i]]}({texts_en[index[0, i]]}): {sorted_probs[0, i]*100:0.1f}%')

    clip_text.append(input_text)
    clip_cos_sim = np.round(cosine_similarity(vecs_array(clip_text), vecs_array(clip_text)), len(clip_text))
    print(clip_cos_sim)
    clip_cos_list = clip_cos_sim[-1].tolist()           # .tolist()で numpy.ndarray --> list
    clip_cos_list.pop(-1)
    avg = sum(clip_cos_list)/ len(clip_cos_list) * 100      # %表示
    print(f'image:{save_image}, prob:{avg:0.2f}%')
    if(avg >= max_prob):
      max_prob_image = save_image
      max_prob = avg

  except Exception as e:
    # print(e)  # エラー内容の確認
    continue


print(f'\n入力文に合う画像は{max_prob_image}で、その確率は{max_prob:0.2f}%です')
