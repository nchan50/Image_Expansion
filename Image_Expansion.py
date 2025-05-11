import numpy as np
import cv2 as cv
import os

# Constants for color channels
BLUE, GREEN, RED = 0, 1, 2
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False

def img_SVD(file, channel):
  u, s, v = np.linalg.svd(cv.imread(file)[:, :, channel], full_matrices = False)
  # We append to for time complexity compared to delete()
  U, S, V = list(), list(), list()
  for c in range(len(s)):
    if s[c] > 1e-8:
      U.append(u[:, c])
      S.append(s[c])
      V.append(v[c, :])
  return np.array(U), np.array(S), np.array(V) # Turn list into arrays
      
def store_USV(U, S, V, directory, name):
  path = f'Database\\{directory}'
  os.makedirs(path, exist_ok=True) # Creates directoryectory
  if not os.path.exists(f'{path}\\{name}.npz') or WRITE_OVER_FILES:
    np.savez(f'{path}\\{name}', U = U, S = S, V = V) # Saves the U, S, and V arrays in a .npz file
  else:
    print(f'{path}\\{name}.npz already exists.')

def store_string(arr, directory, name):
  path = f'Database\\{directory}\\str'
  os.makedirs(path, exist_ok=True) # Creates directoryectory
  if not os.path.exists(f'{path}\\{name}') or WRITE_OVER_FILES:
    np.savetxt(f'{path}\\{name}', arr, fmt='%.2f') # Saves an array in a text file
  else:
    print(f'{path}\\{name} already exists.')
    
def get_USV(name, channel, usv = 'USV'):
  # usv: Letters determine which arrays are returned 
  file = f'Database\\{name}\\{channel}.npz'
  if os.path.isfile(file):
    arrs = list()
    for i in ['U', 'S', 'V']:
      if i in usv:
        arrs.append(np.load(file)[i])
    return tuple(arrs)
  else:
    print(f'The file {file} does not exist.')
  
def create_data(file, c = 'BGR', usv = True, string = False):
  # c : Letters determine the channels selected
  # usv : Determines if USV values are stored as a .npz file
  # str : Determines if USV values are stored as a text file
  name = file[:file.index('.')]
  path = f'Patterns\\{file}'
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    if color[0] in c:
      U, S, V = img_SVD(path, channel)
      if usv: 
        store_USV(U, S, V, name, color)
      if string: 
        store_string(U, name, f'{color}_U')
        store_string(S, name, f'{color}_S')
        store_string(V, name, f'{color}_V')
        
create_data('pattern_1.png', c = 'RGB', string = True)
create_data('pattern_2.png', c = 'RGB', string = True)
U, S, V = get_USV('pattern_2', 'GREEN')
print(S)