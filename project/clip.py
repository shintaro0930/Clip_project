# import os: UNIXコマンドをpythonファイルの中で実行できる
import os
# import sys: モジュール検索パスをできるようにする(他のdirから.pyのクラスなどを取得)
import sys
# sys.path.append('ディレクトリ名'): ディレクトリの名前を検索パスに追加
sys.path.append('CLIP')
import torch
from CLIP import clip
from PIL import Image
# PILの説明: https://note.nkmk.me/python-pillow-basic/
from googletrans import Translator

from pathlib import Path
import pyheif
import glob
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def heic_png(image_path, save_path):
    # HEICファイルpyheifで読み込み
    heif_file = pyheif.read(image_path)
    # 読み込んだファイルの中身をdata変数へ
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    # JPEGで保存
    data.save(str(save_path), "JPEG")

# file_base_dir = '/work/pictures/'
file_base_dir = '/work/new_pictures/'
texts_jp = ["車が写っている晴天の日", "大きな建物", "木とガラスが写った美術館", "様々な色のガラス"]


# ディレクトリ内の画像ファイルを順にリストimagesに追加
images = []
files = os.listdir(file_base_dir)

# .heic, .HEICを消し去りたい
for file in files:
  root_extenstion_tuple = os.path.splitext(file) # root_extension_tuple: tuple型
  if(root_extenstion_tuple[1] == '.heic' or root_extenstion_tuple[1] == '.HEIC'):
    # os.rename(file, root_extenstion_tuple[0] + '.jpg')
    
    # before_image = str(file)
    # after_image = root_extenstion_tuple[0] + '.jpg'
    # heic_png(before_image, after_image)
    continue
  elif(root_extenstion_tuple[1] == '.sh'):
    continue
  images.append(file)
   # print(f"images: {images}") 


# 多言語に翻訳
translator = Translator()
# dest:翻訳先の言語(destination), src:翻訳元の言語(source)
texts_en = [translator.translate(text_jp, dest="en", src="ja").text for text_jp in texts_jp]
# print(texts_en)

for i, image in enumerate(images):
  print(f'--- {images[i]} ---')
  try:
    original_image = Image.open(file_base_dir+f"{image}")
    #画像の前処理
    image = preprocess(original_image).unsqueeze(0).to(device)

    #tokenize
    text = clip.tokenize(texts_en).to(device)
  
    with torch.no_grad():
        #エンコード
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        #推論
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 結果表示 
    for i in range(probs.shape[-1]):
        print(f'{texts_jp[i]}({texts_en[i]}): {probs[0, i]*100:0.1f}%')

    # 例外処理        
  except Exception as e:
    print(e)    # エラーの出力
    continue
