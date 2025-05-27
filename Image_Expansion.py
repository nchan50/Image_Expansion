import numpy as np  # Linear Algebraic Functions
import cv2 as cv  # Image Comprehension
import os  # Manages directories
import pickle  # Stores objects in a file
from scipy.spatial import KDTree  # Used for organizing tuple pairs

from AdjTM import AdjTM

BLUE, GREEN, RED = 0, 1, 2  # Constants for color channels
BGR = ['BLUE', 'GREEN', 'RED']
WRITE_OVER_FILES = False
PRINT_LOG = False
DECIMALS = 10


def img_SVD(file, channel):
    u, s, v = np.linalg.svd(cv.imread(file)[:, :, channel], full_matrices=False)
    # We append to for time complexity compared to delete()
    U, S, V = list(), list(), list()
    for c in range(len(s)):
        if s[c] > 1e-8:
            U.append(u[:, c])
            S.append(s[c])
            V.append(v[c, :])
    return np.array(U).T, np.array(S), np.array(V)  # Turn list into arrays


def store_USV(U, S, V, directory, name):
    path = f'Database\\{directory}'
    os.makedirs(path, exist_ok=True)  # Creates directory
    if not os.path.exists(f'{path}\\{name}.npz') or WRITE_OVER_FILES:
        np.savez(
            f'{path}\\{name}', U=U, S=S, V=V
        )  # Saves the U, S, and V arrays in a .npz file
    elif PRINT_LOG:
        print(f'{path}\\{name}.npz already exists.')


def store_string(arr, directory, name):
    path = f'Database\\{directory}\\str'
    os.makedirs(path, exist_ok=True)  # Creates directory
    if not os.path.exists(f'{path}\\{name}.txt') or WRITE_OVER_FILES:
        np.savetxt(
            f'{path}\\{name}.txt', arr, fmt='%.2f'
        )  # Saves an array in a text file
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
    os.makedirs(path, exist_ok=True)  # Creates directory
    if not os.path.exists(f'{path}\\{name}'):
        with open(f'{path}\\{name}', 'wb') as f:
            pickle.dump(AdjTM(vector), f)  # Saves AdjTM object in a pkl file
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


def get_USV(name, channel, usv='USV'):
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
    file = (
        f'Database\\TM\\{BGR[channel]}\\{orientation}\\pair_{min(pair)}_{max(pair)}.pkl'
    )
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


def create_data(file, c='BGR', usv=True, string=False):
    # c : Letters determine the channels selected
    # usv : Determines if USV values are stored as a .npz file
    # str : Determines if USV values are stored as a text file
    path = f'Patterns\\{file}'
    name = file[: file.index('.')]
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
    name = file[: file.index('.')]
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


def expand_image(file, pixels, sides='L'):
    # sides: Determines which sides to extend. left(L), right(R), top(T), bottom(B)

    RANDOM = False  # RANDOM determines if extending the image is based on random probability or highest probability
    # LRTB is a dictionary containing metadata about side-dependent values and operations
    LRTB = {
        'L': [
            0,
            'U',
            1,
            lambda UV, p: np.hstack((np.zeros((UV[1].shape[0], p)), UV[1])),
            lambda UV, i: UV[0][:, i],
            lambda UV, i, p: list(UV[1][i, :].flatten())[p],
            lambda UV, i, j, e: UV[1].__setitem__((i, j), e),
        ],
        'R': [
            0,
            'U',
            1,
            lambda UV, p: np.hstack((UV[1], np.zeros((UV[1].shape[0], p)))),
            lambda UV, i: UV[0][:, i],
            lambda UV, i, p: list(UV[1][i, :].flatten())[-(p + 1)],
            lambda UV, i, j, e: UV[1].__setitem__((i, -(j + 1)), e),
        ],
        'T': [
            0,
            'V',
            0,
            lambda UV, p: np.vstack((np.zeros((p, UV[0].shape[1])), UV[0])),
            lambda UV, i: UV[1][i, :],
            lambda UV, i, p: list(UV[0][:, i].flatten())[p],
            lambda UV, i, j, e: UV[0].__setitem__((j, i), e),
        ],
        'B': [
            0,
            'V',
            0,
            lambda UV, p: np.vstack((UV[0], np.zeros((p, UV[0].shape[1])))),
            lambda UV, i: UV[1][i, :],
            lambda UV, i, p: list(UV[0][:, i].flatten())[-(p + 1)],
            lambda UV, i, j, e: UV[0].__setitem__((-(j + 1), i), e),
        ],
    }

    # Links pixel arguments to LRTB
    if isinstance(pixels, (list, np.ndarray, set, tuple)):
        if len(pixels) != len(sides):
            raise ValueError('Invalid expansion')
        else:
            for i, char in enumerate(sides):
                LRTB[char][0] = pixels[i]
    else:
        LRTB[sides][0] = pixels

    # Creates extension on each channel
    BGR_UV = list()
    for channel in [BLUE, GREEN, RED]:
        U, S, V = img_SVD(f'Input\\{file}', channel)
        S_sqrt = np.diag([n**0.5 for n in S])
        BGR_UV.append(
            [np.round(U @ S_sqrt, DECIMALS), np.round(S_sqrt @ V, DECIMALS)]
        )  # We consider the square root of diagonal entries to preserve the original image
        UV = BGR_UV[channel]
        for side, values in LRTB.items():
            p = values[0]
            if p == 0:
                continue
            orientation = values[1]
            o = values[2]
            UV[o] = values[3](UV, p)
            # KDTree Documentation: https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.spatial.KDTree.html
            tree = KDTree(list(get_pairs(channel, orientation)))
            for i in range(UV[o].shape[1 - o]):
                oriented_vector = values[4](UV, i)
                tree_queries = [
                    tree.query(
                        (
                            min(oriented_vector[j], oriented_vector[j + 1]),
                            max(oriented_vector[j], oriented_vector[j + 1]),
                        )
                    )
                    for j in range(len(oriented_vector) - 1)
                ]  # Set of indices and distance of closest point in the KDTree to the pair
                # !!! Should I factor in distance
                best_pairs = [tree.data[t[1]] for t in tree_queries]
                TM = get_TM(channel, orientation)
                # We use symmetric normalization to get TM_probs. We assume that unoriented mirror images exist, and so the paths are undirected.
                TM_probs = [
                    TM.get_entry(nodeA, nodeB)
                    / (np.sum(TM.get_row(nodeA)) * np.sum(TM.get_column(nodeB))) ** 0.5
                    for nodeA, nodeB in best_pairs
                ]
                TM_probs /= np.sum(TM_probs)

                # Constructing the best transition matrix to extend the image
                aggregate_TM = AdjTM([])
                for j in range(len(best_pairs)):
                    aggregate_TM += (
                        get_pair_TM(best_pairs[j], channel, orientation).stochastic()
                        * TM_probs[j]
                    )

                # To predict the nth adjacent pixel, we raise our transition matrix to the nth power. We use matrix multiplication to save time complexity.
                exp_aggregate_TM = AdjTM.copy(aggregate_TM)
                for j in reversed(range(p)):
                    key_arr = np.array(list(aggregate_TM.index_map.keys()))
                    closest_value = key_arr[
                        np.abs(key_arr - values[5](UV, i, p)).argmin()
                    ]
                    entry_probs = exp_aggregate_TM.get_column(closest_value)
                    if RANDOM:
                        entry = np.random.choice(
                            list(exp_aggregate_TM.index_map.keys()), p=entry_probs
                        )  # Random Probability Choice
                    else:
                        entry = exp_aggregate_TM.get_node(
                            np.argmax(entry_probs)
                        )  # Highest Probability Choice
                    if j == 0 or j == 3:
                        print(entry)
                    values[6](UV, i, j, entry)
                    exp_aggregate_TM *= aggregate_TM
    return [np.clip(u @ v, 0, 255) for u, v in BGR_UV]


def display_image(img):
    screen_width, screen_height = 1500, 750  # Default screen dimensions
    img = img.astype(np.uint8)  # Makes image compatible with OpenCV

    # OpenCV Documentation: https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
    img = cv.resize(
        img, (screen_width, screen_height), interpolation=cv.INTER_NEAREST_EXACT
    )  # Resizes image
    cv.imshow('Image', img)  # Shows image
    cv.waitKey(0)  # Displays image indefinitely


def delete_dir_recursive(path):
    if not os.path.exists(path):
        return
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            delete_dir_recursive(filepath)
        else:
            os.remove(filepath)
    os.rmdir(path)


# delete_dir_recursive('Database')

# create_data('pattern_2.png', string = True)
# create_TM('pattern_2.png')

# create_data('pattern_3.png', string = True)
# create_TM('pattern_3.png')
# create_data('pattern_4.png', string = True)
# create_TM('pattern_4.png')
# create_data('pattern_5.png', string = True)
# create_TM('pattern_5.png')
# create_data('pattern_6.png', string = True)
# create_TM('pattern_6.png')

create_data('pattern_1.png', string=True)
create_TM('pattern_1.png')

expand_image('pattern_1.png', [4, 0, 0, 0], 'LRBT')
# print(expand_image('pattern_1.png', [4, 0, 0, 0], 'LRBT')[0])
# display_image(cv.merge(expand_image('pattern_1.png', [4, 0, 0, 0], 'LRBT')))
