import os
import sys
sys.path.append('CLIP')
import torch
from CLIP import clip
import matplotlib.pyplot as plt
from PIL import Image
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)


monsters = ['スライム', 'キングスライム', 'メタルスライム', 'ドラキー', 'アームライオン']
self_txt = ['Slime', 'King-slime', 'Metal-slime', 'Drakey', 'Arm-lion']
texts_en = ["a slime", "a king slime", "a metal slime", "a metal king", "a drakey", "an arm lion"]


file_base_dir = '/work/Monster/' 
for i, monster in enumerate(self_txt):
  print(f'--- {monsters[i]} ---')
  original_image = Image.open(file_base_dir+f"{monster}.png")
  image = preprocess(original_image).unsqueeze(0).to(device)
  texts = clip.tokenize(texts_en).to(device)
 

  with torch.no_grad():
      # encode
      image_features = model.encode_image(image)
      text_features = model.encode_text(texts)
       
      # 推論
      logits_per_image, logits_per_text = model(image, texts)
      probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
  for i in range(probs.shape[-1]):
    print(f'{texts_en[i]}: {probs[0, i]*100:0.1f}%')