import os
from pathlib import Path

directory = '/work/pictures_2/'

for f in Path(directory).rglob('*'):
   f.rename(directory + f.stem + '.png')