from PIL import Image
import pyheif
import os

images = []

def change_extensions(files):
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