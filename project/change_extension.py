from PIL import Image
import pyheif
import os


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


images = []
# image_base_dir = '/work/pictures/'
# files = os.listdir('/work/pictures/')

image_base_dir = '/work/sub_pictures/'
files = os.listdir('/work/sub_pictures/')

for file in files:
   full_path_file = image_base_dir + file
   root_extenstion_tuple = os.path.splitext(file)
   if(root_extenstion_tuple[1] == '.heic' or root_extenstion_tuple[1] == '.HEIC'): 
      before_image = str(full_path_file)
      after_image = image_base_dir + root_extenstion_tuple[0] + '.jpg'
      print(f'{before_image}')
      if(os.path.isfile(after_image) == True):
         pass
      else:
         heic_png(before_image, after_image)
      try:
         os.remove(before_image)
      except Exception as e:
         continue
      images.append(file)   