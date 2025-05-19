import numpy as np
import cv2 as cv
import os
import pickle

from AdjTM import AdjTM

# Constants for color channels
BLUE, GREEN, RED = 0, 1, 2
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False
DECIMALS = 3

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
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}.npz') or WRITE_OVER_FILES:
    np.savez(f'{path}\\{name}', U = U, S = S, V = V) # Saves the U, S, and V arrays in a .npz file
  else:
    print(f'{path}\\{name}.npz already exists.')

def store_string(arr, directory, name):
  path = f'Database\\{directory}\\str'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}') or WRITE_OVER_FILES:
    np.savetxt(f'{path}\\{name}', arr, fmt='%.2f') # Saves an array in a text file
  else:
    print(f'{path}\\{name} already exists.')
    
def store_adjTM(vector, directory, name):
  path = f'Database\\TM\\{directory}'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}.pkl'):
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(AdjTM(vector), f) # Saves AdjTM object in a pkl file
  else:
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(pickle.load(f).add_vector(vector), f)
    
    
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
    
  def get_adjTM(pair, orientation, channel):
    
  
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
        
def create_TM(file):
  name = file[:file.index('.')]
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    U, S, V = get_USV(file[:file.index('.')], color)
    for i in range(S.shape[0]):
      u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
      v = np.round(V[i, :] * S[i] ** 0.5, DECIMALS)
      for j in range(len(v) - 1):
        store_adjTM(u, f'{color}\\U', f'pair_{min(v[j], v[j + 1])}_{max(v[j], v[j + 1])}')
      for j in range(len(u) - 1):
        store_adjTM(v, f'{color}\\V', f'pair_{min(u[j], u[j + 1])}_{max(u[j], u[j + 1])}')
        
def expand_image(file, pixels, side = 'r')
  name = file[:file.index('.')]
  size = cv.imread(file).shape
  new_arr = [np.zeros(size[0], size[1] + pixels), np.zeros(size[0], size[1] + pixels), np.zeros(size[0], size[1] + pixels)]
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    U, S, V = get_USV(file[:file.index('.')], color)
    
    new_arr[channel][pixels:, :] = V
    
    
# create_data('pattern_1.png', string = True)
# create_TM('pattern_1.png')
