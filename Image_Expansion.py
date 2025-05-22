import numpy as np # Linear Algebraic Functions
import cv2 as cv # Image Comprehension
import os # Manages directories
import pickle # Stores objects in a file
from scipy.spatial import KDTree # Used for organizing tuple pairs

from AdjTM import AdjTM

BLUE, GREEN, RED = 0, 1, 2 # Constants for color channels
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False
PRINT_LOG = True
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
    file = f'Database\\TM\\{BGR[channel]}\\{orientation}\\pair_{min(pair)}_{max(pair)}'
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

def expand_image(file, pixels, side = 'r'):
    RANDOM = False
    bgr = list()
    for channel in [BLUE, GREEN, RED]:
        U, S, V = img_SVD(f'Input\\{file}', channel)
        bgr.append(np.zeros((len(U), len(V[0]) + pixels)))
        tree = KDTree(list(get_pairs(channel, 'U')))
        for i in range(len(S)):
            u = np.round(U[:, i] * S[i] ** 0.5, DECIMALS)
            v = [0] * pixels
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
                closest_value = key_arr[np.abs(key_arr - V[i, 0] * S[i] ** 0.5).argmin()]
                entry_probs = exp_aggregate_TM.get_column(closest_value)
                if RANDOM:
                    v[-j] = np.random.choice(exp_aggregate_TM.index_map, p = entry_probs)
                else:
                    v[-j] = exp_aggregate_TM.get_node(np.argmax(entry_probs))
                exp_aggregate_TM *= aggregate_TM
            bgr[channel] += np.array(U[:, i] * S[i] ** 0.5).reshape(-1, 1) @ np.array(v + list(V[i, :] * S[i] ** 0.5)).reshape(1, -1)
    return bgr  

create_data('pattern_2.png', string = True)
create_TM('pattern_2.png')

#bgr_image = cv.merge(expand_image('pattern_2.png', 3))

# Clip to valid 8-bit range
#bgr_image = np.clip(bgr_image, 0, 255)

# Convert to uint8 for OpenCV
#bgr_image = bgr_image.astype(np.uint8)

# Get screen size (example for 1080p, adjust as needed or use pyautogui to detect it)
#screen_width, screen_height = 1920, 1080  # or use actual screen size

# Resize image
#resized_image = cv.resize(bgr_image, (screen_width, screen_height), interpolation=cv.INTER_NEAREST)

# Show fullscreen
#cv.namedWindow("Image", cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty("Image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
#cv.imshow("Image", resized_image)
#cv.waitKey(0)
#cv.destroyAllWindows()
