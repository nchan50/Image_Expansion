import numpy as np # Linear Algebraic Functions
import cv2 as cv # Image Comprehension
import os # Manages directories
import pickle # Stores objects in a file
from scipy.spatial import KDTree # Used for organizing tuple pairs

from AdjTM import AdjTM

BLUE, GREEN, RED = 0, 1, 2 # Constants for color channels
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False
PRINT_LOG = False
DECIMALS = 10

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
    if not os.path.exists(f'{path}\\{name}.txt') or WRITE_OVER_FILES:
        np.savetxt(f'{path}\\{name}.txt', arr, fmt='%.2f') # Saves an array in a text file
    elif PRINT_LOG:
        print(f'{path}\\{name}.txt already exists.')

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
    if not os.path.exists(f'{path}\\{name}'):
        with open(f'{path}\\{name}', 'wb') as f:
            pickle.dump(AdjTM(vector), f) # Saves AdjTM object in a pkl file
    else:
        with open(f'{path}\\{name}', 'rb') as f:
            TM = pickle.load(f)
        TM.add_vector(vector)
        with open(f'{path}\\{name}', 'wb') as f:
            pickle.dump(TM, f)
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

def remove_USV(directory, name):
    file = f'Database\\{directory}\\{name}.npz'
    try:
        os.remove(file)
    except FileNotFoundError and PRINT_LOG:
        print(f'{file} does not exists.')

def remove_string(directory, name):
    path = f'Database\\{directory}\\str'
    try:
        os.remove(f'{path}\\{name}.txt')
        if os.path.isdir(path) and not os.listdir(path):
          os.rmdir(path)
    except FileNotFoundError and PRINT_LOG:
        print(f'{file} does not exists.')

def remove_TM(vector, channel, orientation):
    file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\transition.pkl'
    try:
        with open(file, 'rb') as f:
            TM = pickle.load(f)
    except FileNotFoundError and PRINT_LOG:
        print(f'{file} does not exists.')
        return
    TM.remove_vector(vector)
    if TM.TM.shape[0] != 0:
        with open(file, 'wb') as f:
            pickle.dump(TM, f)
    else:
        os.remove(file)

def remove_pair_TM(vector, channel, orientation, pair):
    path = f'Database\\TM\\{BGR[channel]}\\{orientation}'
    name = f'pair_{min(pair)}_{max(pair)}.pkl'
    try:
        with open(f'{path}\\{name}', 'rb') as f:
            TM = pickle.load(f)
    except FileNotFoundError and PRINT_LOG:
        print(f'{path}\\{name} does not exists.')
    TM.remove_vector(vector)
    if TM.TM.shape[0] != 0:
        with open(f'{path}\\{name}', 'wb') as f:
            pickle.dump(TM, f)
    else:
        os.remove(f'{path}\\{name}')
        pairs.remove((min(pair), max(pair)))
    try:
        with open(f'{path}\\pairs.pkl', 'rb') as f:
            pairs = pickle.load(f)
    except FileNotFoundError and PRINT_LOG:
        print(f'{path}\\pairs.pkl does not exist')
    if len(pairs) != 0:
        with open(f'{path}\\pairs.pkl', 'wb') as f:
            pickle.dump(pairs, f)
    else:
        os.remove(f'{path}\\pairs.pkl')

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
        print(f'The file {file}.pkl does not exist.')

def get_pair_TM(pair, channel, orientation):
    file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\pair_{min(pair)}_{max(pair)}.pkl'
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    elif PRINT_LOG:
        print(f'The file {file}.pkl does not exist.')

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
    path = f'Patterns\\{file}'
    name = file[:file.index('.')]
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

def delete_data(file, c = 'BGR', usv = True, string = False):
    # c : Letters determine the channels selected
    # usv : Determines if USV values npz files are deleted
    # str : Determines if USV values txt files are are deleted
    name = file[:file.index('.')]
    for channel in [BLUE, GREEN, RED]:
        color = BGR[channel]
        if color[0] in c:
            if usv: 
                remove_USV(name, color)
            if string: 
                remove_string(name, f'{color}_U')
                remove_string(name, f'{color}_S')
                remove_string(name, f'{color}_V')
    if os.path.isdir(f'Database\\{name}') and not os.listdir(f'Database\\{name}'):
      os.rmdir(f'Database\\{name}')
    

def delete_TM(file):
    name = file[:file.index('.')]
    for channel in [BLUE, GREEN, RED]:
        if os.path.isfile(f'Database\\{name}\\{BGR[channel]}.npz'):
          U, S, V = get_USV(name, channel)
        else:
          U, S, V = img_SVD(f'Patterns\\{file}', channel)
        for i in range(S.shape[0]):
            u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
            v = np.round(V[i, :] * S[i] ** 0.5, DECIMALS)
            for j in range(len(u) - 1):
                remove_pair_TM(v, channel, 'U', (u[j], u[j + 1]))
            for j in range(len(v) - 1):
                remove_pair_TM(u, channel, 'V', (v[j], v[j + 1]))
            remove_TM(u, channel, 'U')
            remove_TM(v, channel, 'V')

def expand_image(file, pixels, sides = 'L'):
    # side: Determine which sides to extend. left(L), right(R), top(T), bottom(B)
    RANDOM = False
    LRTB = dict()
    if isinstance(pixels, (list, np.ndarray, set, tuple)):
      if len(pixels) != len(sides):
        raise ValueError('Invalid expansion')
      else:
        for i, char in enumerate(sides):
          LRTB[char] = pixels[i]
    else:
      LRTB[sides] = pixels
    BGR_UV = list()
    for channel in [BLUE, GREEN, RED]:
        U, S, V = img_SVD(f'Input\\{file}', channel)
        S_sqrt = np.diag([n ** 0.5 for n in S])
        BGR_UV.append([np.round(U @ S_sqrt, DECIMALS), np.round(S_sqrt @ V, DECIMALS)])
        UV = BGR_UV[channel]
        for side, p in LRTB.items():
          if side in ('L', 'R'):
            if side == 'L':
                UV[1] = np.hstack((np.zeros((UV[1].shape[0], p)), UV[1]))
            if side == 'R':
                UV[1] = np.hstack((UV[1], np.zeros((UV[1].shape[0], p))))
            orientation = 'U'
            s = UV[1].shape[0]
          if side in ('T', 'B'):
            if side == 'T':
                UV[0] = np.vstack((np.zeros((p, UV[0].shape[1])), UV[0]))
            if side == 'B':
                UV[0] = np.vstack((UV[0], np.zeros((p, UV[0].shape[1]))))
            orientation = 'V'
            s = UV[0].shape[1]
          tree = KDTree(list(get_pairs(channel, orientation)))
          for i in range(s):
              if orientation == 'U':
                oriented_vector = UV[0][:, i]
              if orientation == 'V':
                oriented_vector = UV[1][i, :]
              tree_queries = [tree.query((min(oriented_vector[j], oriented_vector[j + 1]), max(oriented_vector[j], oriented_vector[j + 1]))) for j in range(len(oriented_vector) - 1)] # !!! Should I factor in distance
              best_pairs = [tree.data[t[1]] for t in tree_queries]
              TM = get_TM(channel, orientation)
              TM_probs = [TM.get_entry(nodeA, nodeB)/ (np.sum(TM.get_row(nodeA)) * np.sum(TM.get_column(nodeB))) ** 0.5 for nodeA, nodeB in best_pairs] # Uses Symmetric Normalization
              TM_probs /= np.sum(TM_probs)
              aggregate_TM = AdjTM([])
              for j in range(len(best_pairs)):
                  aggregate_TM += get_pair_TM(best_pairs[j], channel, orientation).stochastic() * TM_probs[j]
              exp_aggregate_TM = AdjTM.copy(aggregate_TM) # Matrix multiplication faster than matrix exponentiation
              for j in reversed(range(p)):
                key_arr = np.array(list(aggregate_TM.index_map.keys()))
                if side == 'L':
                  closest_value = key_arr[np.abs(key_arr - list(UV[1][i, :].flatten())[p]).argmin()]
                if side == 'R':
                  closest_value = key_arr[np.abs(key_arr - list(UV[1][i, :].flatten())[-(p + 1)]).argmin()]
                if side == 'T':
                  closest_value = key_arr[np.abs(key_arr - list(UV[0][:, i].flatten())[p]).argmin()]
                if side == 'B':
                  closest_value = key_arr[np.abs(key_arr - list(UV[0][:, i].flatten())[-(p + 1)]).argmin()]
                entry_probs = exp_aggregate_TM.get_column(closest_value)
                if RANDOM:
                  entry = np.random.choice(exp_aggregate_TM.index_map, p = entry_probs)
                else:
                  entry = exp_aggregate_TM.get_node(np.argmax(entry_probs))
                if side == 'L':
                    UV[1][i, j] = entry
                if side == 'R':
                    UV[1][i, -(j + 1)]= entry
                if side == 'T':
                    UV[0][j, i]= entry
                if side == 'B':
                    UV[0][-(j + 1), i]= entry
                exp_aggregate_TM *= aggregate_TM
    return [u @ v for u, v in BGR_UV]

def display_image(img):
  img = np.clip(img, 0, 255)

  img = img.astype(np.uint8)
  screen_width, screen_height = 1920, 1080
  img = cv.resize(img, (screen_width, screen_height), interpolation=cv.INTER_NEAREST)

  cv.namedWindow("Image", cv.WND_PROP_FULLSCREEN)
  cv.setWindowProperty("Image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
  cv.imshow("Image", img)
  cv.waitKey(0)
  cv.destroyAllWindows()
  
create_data('pattern_2.png', string = True)
create_TM('pattern_2.png')

display_image(cv.merge(expand_image('pattern_2.png', [2, 4, 5, 2], 'LRBT')))
