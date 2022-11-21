import os
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

# text_base_dir = '/work/texts/'
text_base_dir = '/work/project/texts/'
# image_base_dir = '/work/project/pictures/'
# image_base_dir = '/work/light_pictures/'
image_base_dir = '/work/project/light_pictures/'




texts_jp = []
texts_dir = os.listdir(text_base_dir)

text_file = text_base_dir + 'text.txt'
# text_file = text_base_dir + 'rand_text.txt'

with open(text_file) as texts:
  for text in texts:
    texts_jp.append(text.rstrip())    #.rstrip()は改行コードを消す


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

for i, image in enumerate(images):
  try:
    original_image = Image.open(image_base_dir + f"{image}")
    print(f'--- {images[i]} ---')
    image = preprocess(original_image).unsqueeze(0).to(device)
    text = clip.tokenize(texts_en).to(device)
  
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        sorted_probs = np.sort(probs)
        index = np.argsort(probs)

    # print("=== NORMAL===")
    # for i in range(probs.shape[-1]):
    #     print(f'{texts_jp[i]}({texts_en[i]}): {probs[0, i]*100:0.1f}%')
    
    for i in reversed(range(sorted_probs.shape[-1] - 3, sorted_probs.shape[-1])):
        print(f'{texts_jp[index[0, i]]}({texts_en[index[0, i]]}): {sorted_probs[0, i]*100:0.1f}%')
    
    print("\n")

  except Exception as e:
    print(e)
    continue