import os
import re
import shutil
import sys
sys.path.append('CLIP')
import torch
from CLIP import clip
from PIL import Image
from googletrans import Translator
import numpy as np 


# choose the device 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# initialize base directory
text_base_dir = '/work/texts/'
image_base_dir = '/work/pictures/'


# remove '\r' or '\r\n' and punctuations
def remove_punctuation(input):
  output = re.sub(r'[^\w\s]','',input)
  output = output.rstrip()
  return output


# initialize text lists and designate the full path
texts_jp:list = []
texts_dir = os.listdir(text_base_dir)
text_file = text_base_dir + 'text.txt'

with open(text_file) as texts:
  for text in texts:
    texts_jp.append(remove_punctuation(text)) 

images = []
files = os.listdir(image_base_dir)
for file in files:
  try:
    images.append(file)
  except Exception as e:
    continue

translator = Translator()
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]


# initialize
clip_text:list = []
clip_cos_list = []
max_prob = 0
max_prob_image = images[0]
save_images:list = []

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
    
    # ソートした上位3つのテキスト文章をlistに追加 
    with open('output.txt', 'a', encoding='utf-8', newline='\n') as f:
      f.write('=============\n')
      f.write(f'{save_image}\n')
      print("=============")
      print(save_image)
      for i in reversed(range(sorted_probs.shape[-1] - 3, sorted_probs.shape[-1])):
          clip_text.append(texts_jp[index[0, i]])
          f.write(f'{texts_jp[index[0, i]]}({texts_en[index[0, i]]}): {sorted_probs[0, i]*100:0.1f}%\n')

          ## 写真にマッチするテキストとその確率
          print(f'{texts_jp[index[0, i]]}({texts_en[index[0, i]]}): {sorted_probs[0, i]*100:0.1f}%')
      shutil.move(image_base_dir + save_image, '/work/used_pictures/')

  except Exception as e:
    print(e)
    continue