import numpy as np # Linear Algebraic Functions
import cv2 as cv # Image Comprehension
import os # Manages directories
import pickle # Stores objects in a file
from scipy.spatial import KDTree # Used for organizing tuple pairs

from AdjTM import AdjTM

BLUE, GREEN, RED = 0, 1, 2 # Constants for color channels
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False
DECIMALS = 3

# Creates pkl file to store ordered tuple pairs
for color in BGR:
  path = f'Database\\TM\\{color}'
  os.makedirs(f'{path}\\U', exist_ok = True)
  if not os.path.exists(f'{path}\\U\\pairs.pkl')
    with open(f'{path}\\U\\pairs', 'wb') as f:
      pickle.dump(set(), f)
  os.makedirs(f'{path}\\V', exist_ok = True)
  if not os.path.exists(f'{path}\\V\\pairs.pkl')
    with open(f'{path}\\V\\pairs', 'wb') as f:
      pickle.dump(set(), f)
  
# Creates pkl file to store transition matrix
for color in BGR:
  path = f'Database\\TM\\{color}'
  os.makedirs(f'{path}\\U', exist_ok = True)
  if not os.path.exists(f'{path}\\U\\transition.pkl')
    with open(f'{path}\\U\\transition', 'wb') as f:
      pickle.dump(AdjTM([]), f)
  os.makedirs(f'{path}\\V', exist_ok = True)
  if not os.path.exists(f'{path}\\V\transition.pkl')
    with open(f'{path}\\V\\transition', 'wb') as f:
      pickle.dump(AdjTM([]), f)


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
    
def store_TM(vector, channel, orientation):
  path = f'Database\\TM\\{channel}\\{orientation}'
  with open(f'{path}\\transition.pkl', 'wb') as f:
    pickle.dump(pickle.load(f).add_vector(vector), f)
    
def store_pair_TM(vector, channel, orientation, pair):
  path = f'Database\\TM\\{channel}\\{orientation}'
  name = f'pair_{min(pair)}_{max(pair)}'
  os.makedirs(path, exist_ok = True) # Creates directory
  if not os.path.exists(f'{path}\\{name}.pkl'):
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(AdjTM(vector), f) # Saves AdjTM object in a pkl file
  else:
    with open(f'{path}\\{name}', 'wb') as f:
      pickle.dump(pickle.load(f).add_vector(vector), f)
  with open(f'{path}\\pairs.pkl', 'wb') as f:
      pickle.dump(pickle.load(f).add((min(pair)_max(pair))), f)
    
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
    
def get_TM(channel, orientation):
  file = f'Database\\TM\\{channel}\\{orientation}\\transition.pkl'
  if os.path.isfile(file):
    with open(file, 'wb') as f:
      return pickle.load(f)
  else:
    print(f'The file {file} does not exist.')
    
def get_pair_TM(pair, channel, orientation):
  file = f'Database\\TM\\{channel}\\{orientation}\\pair_{min(pair)}_{max(pair)}.pkl'
  if os.path.isfile(file):
    with open(file, 'wb') as f:
      return pickle.load(f)
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
        
def create_TM(file):
  name = file[:file.index('.')]
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    U, S, V = get_USV(file[:file.index('.')], color)
    for i in range(S.shape[0]):
      u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
      v = np.round(V[i, :] * S[i] ** 0.5, DECIMALS)
      store_pair_TM(u, color, 'U', (v[j], v[j + 1]))
      for j in range(len(v) - 1):
        store_pair_TM(v, color, 'U', (u[j], u[j + 1]))
      for j in range(len(u) - 1):
        store_pair_TM(u, color, 'V', (v[j], v[j + 1]))
        
def expand_image(file, pixels, side = 'r'):
  RANDOM = True
  name = file[:file.index('.')]
  size = cv.imread(file).shape
  new_channel_arrs = [np.zeros(size[0], size[1] + pixels)] * 3
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    U, S, V = get_USV(file[:file.index('.')], color)
    tree = KDTree(pickle.load(f))
    for i in len(U):
      u = U[:, i] * S[i] ** 0.5
      v = list(len(V[0]) + pixels)
      v[pixels:] = V[i, :] * S[i] ** 0.5
      tree_queries = [tree.query(min(U[j], U[j + 1]), max(U[j], U[j + 1])) for j in range(len(u))] # !!! Should I factor in distance?
      best_pairs = [tree[t[1]] for t in tree_queries]
      tm = get_TM(color, 'U')
      probs = [tm.get_entry(nodeA, nodeB)/ (np.sum(tm.get_row(nodeA)) * np.sum(tm.get_col(nodeB))) ** 0.5 for nodeA, nodeB in best_pairs] # Uses Symmetric Normalization
      probs = probs / np.linalg.norm(probs)
      aggregate_TM = AdjTM([])
      for j in range(len(best_pairs)):
        aggregate_TM = aggregate_TM + get_pair_TM(best_pairs[i], f'{color}\\U') * probabilities[j]
      exponentiated_aggregate_TM = aggregate_TM * 1
      for k in range(pixels):
        probabilities = exponentiated_aggregate_TM.get_column(aggregate_TM.index_map[np.abs(aggregate_TM.index_map - v[pixels]).argmin()])
        probabilities = probabilities / np.linalg.norm(probabilities)
        if RANDOM:
          v[pixels - k] = np.random.choice(range(len(aggregate_TM.index_map)), p=probabilities)
        else:
          v[pixels - k] = aggregate_TM.get_node(np.argmax(probabilities)) 
        exponentiated_aggregate_TM *= aggregate_TM
      new_channel_arrs[channel] += u @ v     
    
  
# create_data('pattern_1.png', string = True)
# create_TM('pattern_1.png')
