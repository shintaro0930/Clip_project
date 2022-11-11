# import os: UNIXコマンドをpythonファイルの中で実行できる
import os
# import sys: モジュール検索パスをできるようにする(他のdirから.pyのクラスなどを取得)
import sys
# sys.path.append('ディレクトリ名'): ディレクトリの名前を検索パスに追加
sys.path.append('CLIP')
import torch
from CLIP import clip
import matplotlib.pyplot as plt
from PIL import Image
from googletrans import Translator
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# これは個別にインストールする必要がある
# !pip install googletrans==4.0.0-rc1
# from googletrans import Translator

file_base_dir = '/work/project/Monster/'
texts_jp = ["スライムに乗った騎士", "馬に乗った騎士", "剣で遊んでいる騎士", "スライム", "スライムに乗った魔法使い"]
monsters = ['Dragon-knight']
translator = Translator()
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]
print(texts_en)
for monster in monsters:
  original_image = Image.open(file_base_dir+f"{monster}.png")
  image = preprocess(original_image).unsqueeze(0).to(device)
  plt.figure()
  plt.imshow(original_image)
  plt.axis('off')
  plt.show()
 
  text = clip.tokenize(texts_en).to(device)
 
  with torch.no_grad():
      image_features = model.encode_image(image)
      text_features = model.encode_text(text)
       
      logits_per_image, logits_per_text = model(image, text)
      probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
  for i in range(probs.shape[-1]):
    print(f'{texts_jp[i]}: {probs[0, i]*100:0.1f}%')