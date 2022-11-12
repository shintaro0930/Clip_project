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

file_base_dir = '/work/pictures/'
# file_base_dir = '/work/project/Monster/'
# texts_jp = ["スライムに乗った騎士", "馬に乗った騎士", "剣で遊んでいる騎士", "スライム", "スライムに乗った魔法使い"]
texts_jp = ["車が写っている晴天の日", "大きな建物", "木とガラスが写った美術館", "様々な色のガラス"]
# monsters = ['Dragon-knight']
monsters = ['IMG_8727', 'IMG_9576', 'IMG_9603', 'IMG_9615', 'IMG_9649', 'IMG_9657']

translator = Translator()
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]
# print(texts_en)

for i, monster in enumerate(monsters):
  try:
    original_image = Image.open(file_base_dir+f"{monster}.png")
    print(f'--- {monsters[i]} ---')
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

    # 結果表示 
    for i in range(probs.shape[-1]):
        print(f'{texts_jp[i]}({texts_en[i]}): {probs[0, i]*100:0.1f}%')

    # 例外処理        
  except Exception as e:
    continue

    # # 結果表示 
    # for i in range(probs.shape[-1]):
    #     print(f'{texts_jp[i]}({texts_en[i]}): {probs[0, i]*100:0.1f}%')