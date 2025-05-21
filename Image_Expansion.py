import numpy as np # Linear Algebraic Functions
import cv2 as cv # Image Comprehension
import os # Manages directories
import pickle # Stores objects in a file
from scipy.spatial import KDTree # Used for organizing tuple pairs

from AdjTM import AdjTM

BLUE, GREEN, RED = 0, 1, 2 # Constants for color channels
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = True
PRINT_LOG = True
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
  return np.array(U).T, np.array(S), np.array(V) # Turn list into arrays
      
def store_USV(U, S, V, directory, name):
  path = f'Database\\{directory}'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}.npz') or WRITE_OVER_FILES:
    np.savez(f'{path}\\{name}', U = U, S = S, V = V) # Saves the U, S, and V arrays in a .npz file
  elif PRINT_LOG:
    print(f'{path}\\{name}.npz already exists.')

def store_string(arr, directory, name):
  path = f'Database\\{directory}\\str'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}') or WRITE_OVER_FILES:
    np.savetxt(f'{path}\\{name}', arr, fmt='%.2f') # Saves an array in a text file
  elif PRINT_LOG:
    print(f'{path}\\{name} already exists.')
    
def store_TM(vector, channel, orientation):
  file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\transition.pkl'
  try:
   with open(file, 'rb') as f:
     TM = pickle.load(f)
  except IOError:
    TM = AdjTM([])
    with open(file, 'wb') as f:
      pickle.dump(AdjTM([]), f)
  TM.add_vector(vector)
  with open(file, 'wb') as f:
    pickle.dump(TM, f)
    
def store_pair_TM(vector, channel, orientation, pair):
  path = f'Database\\TM\\{BGR[channel]}\\{orientation}'
  name = f'pair_{min(pair)}_{max(pair)}.pkl'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}') or WRITE_OVER_FILES:
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(AdjTM(vector), f) # Saves AdjTM object in a pkl file
  else:
    with open(f'{path}\\{name}', 'rb') as f:
      TM = pickle.load(f)
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(TM.add_vector(vector), f)
  try:
    with open(f'{path}\\pairs.pkl', 'rb') as f:
      pairs = pickle.load(f)
  except IOError:
    pairs = set()
    with open(f'{path}\\pairs.pkl', 'wb') as f:
      pickle.dump(pairs, f)
  pairs.add((min(pair), max(pair)))
  with open(f'{path}\\pairs.pkl', 'wb') as f:
    pickle.dump(pairs, f)
    
def get_USV(name, channel, usv = 'USV'):
  # usv: Letters determine which arrays are returned 
  file = f'Database\\{name}\\{BGR[channel]}.npz'
  if os.path.isfile(file):
    arrs = list()
    for i in ['U', 'S', 'V']:
      if i in usv:
        arrs.append(np.load(file)[i])
    return tuple(arrs)
  elif PRINT_LOG:
    print(f'The file {file} does not exist.')
    
def get_TM(channel, orientation):
  file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\transition.pkl'
  if os.path.isfile(file):
    with open(file, 'rb') as f:
      return pickle.load(f)
  elif PRINT_LOG:
    print(f'The file {file} does not exist.')
    
def get_pair_TM(pair, channel, orientation):
  file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\pair_{min(pair)}_{max(pair)}.pkl'
  if os.path.isfile(file):
    with open(file, 'rb') as f:
      return pickle.load(f)
  elif PRINT_LOG:
    print(f'The file {file} does not exist.')
    
def get_pairs(channel, orientation):
  file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\pairs.pkl'
  if os.path.isfile(file):
    with open(file, 'rb') as f:
      return pickle.load(f)
  elif PRINT_LOG:
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
        
def create_TM(file):
  name = file[:file.index('.')]
  for channel in [BLUE, GREEN, RED]:
    U, S, V = get_USV(name, channel)
    for i in range(S.shape[0]):
      u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
      v = np.round(V[i, :] * S[i] ** 0.5, DECIMALS)
      for j in range(len(u) - 1):
        store_pair_TM(v, channel, 'U', (u[j], u[j + 1]))
      for j in range(len(v) - 1):
        store_pair_TM(u, channel, 'V', (v[j], v[j + 1]))
      store_TM(u, channel, 'U')
      store_TM(v, channel, 'V')
        
def expand_image(file, pixels, side = 'r'):
  RANDOM = False
  bgr = []
  for channel in [BLUE, GREEN, RED]:
    U, S, V = img_SVD(f'Input\\{file}', channel)
    bgr.append(np.zeros((len(U), len(V[0]) + pixels)))
    tree = KDTree(list(get_pairs(channel, 'U')))
    for i in range(len(S)):
      u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
      v = np.concatenate(([0] * pixels, np.round(V[i, :] * S[i] ** 0.5, DECIMALS)))
      tree_queries = [tree.query((min(u[j], u[j + 1]), max(u[j], u[j + 1]))) for j in range(len(u) - 1)] # !!! Should I factor in distance
      best_pairs = [tree.data[t[1]] for t in tree_queries]
      TM = get_TM(channel, 'U')
      TM_probs = [TM.get_entry(nodeA, nodeB)/ (np.sum(TM.get_row(nodeA)) * np.sum(TM.get_column(nodeB))) ** 0.5 for nodeA, nodeB in best_pairs] # Uses Symmetric Normalization
      TM_probs /= np.linalg.norm(TM_probs)
      aggregate_TM = AdjTM([])
      for j in range(len(best_pairs)):
        aggregate_TM += get_pair_TM(best_pairs[j], channel, 'U').stochastic() * TM_probs[j] 
      exp_aggregate_TM = AdjTM.copy(aggregate_TM)
      for j in range(pixels):
        key_arr = np.array(list(aggregate_TM.index_map.keys()))
        closest_value = key_arr[np.abs(key_arr - v[pixels]).argmin()]
        entry_probs = exp_aggregate_TM.get_column(closest_value)
        if RANDOM:
          v[pixels - j] = np.random.choice(exp_aggregate_TM.index_map, p = entry_probs)
        else:
          v[pixels - j] = exp_aggregate_TM.get_node(np.argmax(entry_probs))
        exp_aggregate_TM *= aggregate_TM
      bgr[channel] += np.array(u).reshape(-1, 1) @ np.array(v).reshape(1, -1)
  return bgr  

        
create_data('pattern_1.png', string = True)
create_TM('pattern_1.png')

bgr = expand_image('pattern_1.png', 2)
bgr_image = cv.merge(bgr)