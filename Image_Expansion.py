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
      
def store_USV(U, S, V, dir, name):
  path = 'Database\\' + dir
  os.makedirs(path, exist_ok=True) # Creates directory
  if not os.path.exists(path + '\\' + name + '.npz') or WRITE_OVER_FILES:
    np.savez(path + '\\' + name, U_data = U, S_data = S, V_data = V) # Saves the U, S, and V arrays in a .npz file
  else:
    print(path + '\\' + name + '.npz' + " already exists.")

def store_string(arr, dir, name):
  path = 'Database\\' + dir + "\\str"
  os.makedirs(path, exist_ok=True) # Creates directory
  if not os.path.exists(path + '\\' + name) or WRITE_OVER_FILES:
    np.savetxt(path + "\\" + name, arr, fmt='%.2f') # Saves an array in a text file
  else:
    print(path + '\\' + name + " already exists.")
  
def create_data(file, c = 'BGR', usv = True, string = False):
  # c : Letters determine the channels selected
  # usv : Determines if USV values are stored as a .npz file
  # str : Determines if USV values are stored as a text file
  name = file[:file.index('.')]
  path = 'Patterns\\' + file
  for channel in [BLUE, GREEN, RED]:
    color = BGR[channel]
    if color[0] in c:
      U, S, V = img_SVD(path, channel)
      if usv: 
        store_USV(U, S, V, name, color)
      if string: 
        store_string(U, name, color + '_U')
        store_string(S, name, color + '_S')
        store_string(V, name, color + '_V')
        
create_data('pattern_1.png', c = 'RGB', string = True)
create_data('pattern_2.png', c = 'RGB', string = True)