import os # Manage directories and files
import pickle # Store objects in a file
from collections import Counter # Creates a dictionary from a list

import cv2 as cv # Image Comprehension
from numba import njit # Faster looping
import numpy as np # Arrays and Relevant Operations
from numpy import linalg # Specific Linear Algebraic Operations
from scipy.signal.windows import gaussian # Gaussian distributions
from scipy.spatial import KDTree # Used for organizing coordinates

DECIMALS = 4


def img_SVD(img):
    u, s, v = np.linalg.svd(img, full_matrices = False)
    U, S, V = list(), list(), list()
    # We append to for time complexity compared to delete()
    for c in range(len(s)):
        if s[c] > 1e-8:
            U.append(np.round(u[:, c], decimals = DECIMALS))
            S.append(s[c])
            V.append(np.round(v[c, :], decimals = DECIMALS))
    return np.array(U).T, np.array(S), np.array(V)  # Turn list into arrays


def get_all_A(img, width, U, V):
    all_A = dict()
    half = width // 2
    
    # We pad an array on the relevant two sides in order to allow for centers at the edge to have full neighborhoods
    # Feel free to try different padding modes(e.g., 'constant', 'edge', etc.): https://numpy.org/doc/stable/reference/generated/numpy.pad.html 
    padded_U = np.pad(U, ((half, half), (0, 0)), mode = 'reflect')
    padded_V = np.pad(V, ((0, 0), (half, half)), mode = 'reflect')
    for r, c in np.ndindex(img.shape):
        width_U = padded_U[r:r + width, :]
        width_V = padded_V[:, c:c + width]
        A = list()
        for j in range(width ** 2):
            arr = list()
            for i in range(U.shape[1]):
                val = width_U[j // width, i] * width_V[i, j % width]
                arr.append(val)
            A.append(arr)
        A = np.round(A, decimals = DECIMALS).tobytes() # Makes numpy array hashable
        if A not in all_A:
            all_A[A] = list()
        all_A[A].append(img[r, c])
    return all_A
    
    
def img_n_vector(img, width, center, A):
    half = width // 2
    r, c = center
    n_vector = list()

    neighborhood = np.full((width, width), np.nan, dtype = float)
    
    # Bounds of the patch in the image
    r_start = max(r - half, 0)
    r_end = min(r + half + 1, img.shape[0])
    c_start = max(c - half, 0)
    c_end = min(c + half + 1, img.shape[1])
    
    # Start indices in neighborhood
    r_start_neigh = half - (r - r_start)
    c_start_neigh = half - (c - c_start)
    
    # Slices an image patch for neighborhood
    patch = img[r_start:r_end, c_start:c_end]
    neighborhood[r_start_neigh : r_start_neigh + patch.shape[0], c_start_neigh : c_start_neigh + patch.shape[1]] = patch
    
    # Least Squares
    b = neighborhood.ravel() # Equivalent to flatten() but doesn't try making a copy
    b_mask = list()
    A_mask = list()
    for i in range(width ** 2):
        if not np.isnan(b[i]):
            b_mask.append(b[i])
            A_mask.append(A[i])
    if all(i == 0 for i in b_mask) and all(j == 0 for r in A_mask for j in r):
        return np.array([np.nan] * A.shape[1])
    return np.array(linalg.lstsq(A_mask, b_mask, rcond = None)[0])

# Numba jit Documentation: https://numba.pydata.org/numba-doc/dev/user/5minguide.html 
@njit
def distance(a, b, g):
    GSSD = 0
    for i in range(len(a)):
        if not np.isnan(a[i]) and not np.isnan(b[i]):
            GSSD += g[i] * (a[i] - b[i]) ** 2
    return GSSD


def expand_image(file, width, error, pixels, sides = 'L', noise_reduc = 0, uniform = False, samples = None):
    # pixels: individual or list of integers of number of pixels to expand each side by
    # sides: determines which sides to extend. left(L), right(R), top(T), bottom(B)
    # noise_reduc: percentage[0, 1] of noise to remove(proxy for rank reduction)
    # uniform: determines if uniform or gaussian probability distribution is applied to the centers
    # samples: non-parametric or parametric image file(s)
    assert width % 2 == 1
    assert noise_reduc >= 0
    
    os.makedirs('Patterns', exist_ok = True)
    img = cv.imread(f'Patterns\\{file}').astype(float)
    assert width < img.shape[0] and width < img.shape[1]
    
    # Creates the gausian kernel
    # Gaussian Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html 
    gaussian_dist = gaussian(width, 1)
    g = np.outer(gaussian_dist, gaussian_dist).ravel()
    g *= (np.sum(g) - g[len(g) // 2]) / np.sum(g) # Normalizes kernel for ignoring center
        

    bgr_n_vectors = [dict(), dict(), dict()]
    color_vocabulary = set()
    os.makedirs('Samples_v3', exist_ok = True)
    
    # Non-Paraametric Sampling
    if samples == None:
        os.makedirs(f'Samples_v3\\{file}', exist_ok = True)
        color_path = f'Samples_v3\\{file}\\color.pkl'
        if os.path.exists(color_path):
            with open(color_path, 'rb') as f:
                color_vocabulary = pickle.load(f)
        else:
            img_colors = img.reshape(-1, img.shape[2])
            for img_color in img_colors:
                color_vocabulary.add(tuple(img_color))
            with open(color_path, 'wb') as f:
                pickle.dump(color_vocabulary, f)
        for channel in [0, 1, 2]:
            color_n_vectors = bgr_n_vectors[channel]
            sample_path = f'Samples_v3\\{file}\\width_{width}_{channel}_{DECIMALS}.pkl'
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    bgr_n_vectors[channel] = pickle.load(f)
            else:
                U, S, V = img_SVD(img[:, :, channel])
                color_n_vectors[tuple(np.diag(U.T @ img[:, : , channel] @ V.T))] = {(tuple(U.flatten()), tuple(V.flatten())) : get_all_A(img[:, :, channel], width, U, V)}
                os.makedirs(os.path.dirname(sample_path), exist_ok = True)
                with open(sample_path, 'wb') as f:
                        pickle.dump(bgr_n_vectors[channel], f)
    
    # Parametric Sampling by One Image                    
    if isinstance(samples, str):
        os.makedirs(f'Samples_v3\\{samples}', exist_ok = True) 
        color_path = f'Samples_v3\\{samples}\\color.pkl'
        sample_img = cv.imread(f'Patterns\\{sample}').astype(float)
        if os.path.exists(color_path):
            with open(color_path, 'rb') as f:
                color_vocabulary = pickle.load(f)
        else:
            img_colors = sample_img.reshape(-1, sample_img.shape[2])
            for img_color in img_colors:
                color_vocabulary.add(tuple(img_color))
            with open(color_path, 'wb') as f:
                pickle.dump(color_vocabulary, f)
        for channel in [0, 1, 2]:
            color_n_vectors = bgr_n_vectors[channel]
            sample_path = f'Samples_v3\\{samples}\\width_{width}_{channel}_{DECIMALS}.pkl'
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    bgr_n_vectors[channel] = pickle.load(f)
            else:
                U, S, V = img_SVD(sample_img[:, :, channel])
                color_n_vectors[tuple(np.diag(U.T @ sample_img[:, : , channel] @ V.T))] = {(tuple(U.flatten()), tuple(V.flatten())) : get_all_A(sample_img[:, :, channel], width, U, V)}
                os.makedirs(os.path.dirname(sample_path), exist_ok = True)
                with open(sample_path, 'wb') as f:
                        pickle.dump(bgr_n_vectors[channel], f)
    
    # Parametric Sampling by Set of Images                    
    if isinstance(samples, (list, np.ndarray, tuple)):
        samples_path = 'Samples_v3\\'
        samples = sorted(samples)
        for sample in samples:
            os.makedirs(f'Samples_v3\\{sample}', exist_ok = True)
            samples_path += sample
            if len(samples) > 1:
                samples_path += '_'
        os.makedirs(samples_path, exist_ok = True)
        if os.path.exists(samples_path + '\\color.pkl'):
            with open(samples_path + '\\color.pkl', 'rb') as f:
                color_vocabulary = pickle.load(f)
        else:
            for sample in samples:
                sample_img = cv.imread(f'Patterns\\{sample}').astype(float)
                sample_color_path = f'Samples_v3\\{sample}\\color.pkl'
                sample_color_vocabulary = set()
                if os.path.exists(sample_color_path):
                    with open(sample_color_path, 'rb') as f:
                        sample_color_vocabulary = pickle.load(f)
                else:
                    img_colors = sample_img.reshape(-1, sample_img.shape[2])
                    for img_color in img_colors:
                        sample_color_vocabulary.add(tuple(img_color))
                    with open(sample_color_path, 'wb') as f:
                        pickle.dump(sample_color_vocabulary, f)
                color_vocabulary.update(sample_color_vocabulary)
            if len(samples) > 1:
                with open(samples_path + '\\color.pkl', 'wb') as f:
                    pickle.dump(color_vocabulary, f)
        for channel in [0, 1, 2]:
            if os.path.exists(samples_path + f'\\width_{width}_{channel}_{DECIMALS}.pkl'):
                with open(samples_path + f'\\width_{width}_{channel}_{DECIMALS}.pkl', 'rb') as f:
                    bgr_n_vectors[channel] = pickle.load(f)
            else:
                for sample in samples:
                    sample_img = cv.imread(f'Patterns\\{sample}').astype(float)
                    color_n_vectors = bgr_n_vectors[channel]
                    sample_path = f'Samples_v3\\{sample}\\width_{width}_{channel}_{DECIMALS}.pkl'
                    if os.path.exists(sample_path):
                        with open(sample_path, 'rb') as f:
                            sample_color_n_vector = pickle.load(f)
                            color_n_vector = list(sample_color_n_vector.keys())[0]
                            color_UV = list(list(sample_color_n_vector.values())[0].keys())[0]
                            color_As = list(list(sample_color_n_vector.values())[0].values())[0]
                    else:
                        U, S, V = img_SVD(sample_img[:, :, channel])
                        color_n_vector = tuple(np.diag(U.T @ sample_img[:, : , channel] @ V.T))
                        color_UV = tuple(U.flatten()), tuple(V.flatten())
                        color_As = get_all_A(sample_img[:, :, channel], width, U, V)
                        os.makedirs(os.path.dirname(sample_path), exist_ok = True)
                        with open(sample_path, 'wb') as f:
                            pickle.dump({color_n_vector: {color_UV: color_As}}, f)
                    if color_n_vector not in color_n_vectors.keys():
                        color_n_vectors[color_n_vector] = {color_UV: color_As}
                    else:
                        for color_UVs, color_As in color_n_vectors[color_n_vector].items():
                            if color_UV not in color_UVs.keys:
                                color_n_vectors[color_n_vector][color_UV] = color_As
                            else:
                                color_UV_dict = color_n_vectors[color_n_vector][color_UV]
                                color_UV_keys = color_UV_dict.keys()
                                for color_A, color_centers in color_As.items():
                                    if color_A not in color_UV_keys:
                                        color_UV_dict[colorA] = list()
                                    color_UV_dict[colorA].extend(color_centers)
                if len(sample) > 1:
                    with open(samples_path + f'\\width_{width}_{channel}_{DECIMALS}.pkl', 'wb') as f:
                        pickle.dump(bgr_n_vectors[channel], f)
    
    # Allows easier searching for valid colors
    # KDTree requires ordered iterables
    color_vocabulary = list(color_vocabulary)
    color_tree = KDTree(color_vocabulary)
    
    # Dictionary of sides and relevant metadata
    # Metadata:
    # 0: Number of pixels to extend
    # 1: Adding a column or row of all np.nan
    # 2: Distance between img neighborhood coordinate and pixel neighborhood coordinate
    # 3: Changes pixel in image
    LRTB = {
        'L': [
            0,
            lambda img: np.hstack((np.full((img.shape[0], 1, img.shape[2]), np.nan, dtype = float), img)),
            lambda img: img.shape[0],
            lambda img, w, i, A, n, g: distance(img_n_vector(img, w, (i, 0), A), n, g),
            lambda img, i, e: img.__setitem__((i, 0, slice(None)), e),
        ],
        'R': [
            0,
            lambda img: np.hstack((img, np.full((img.shape[0], 1, img.shape[2]), np.nan, dtype = float))),
            lambda img: img.shape[0],
            lambda img, w, i, A, n, g: distance(img_n_vector(img, w, (i, img.shape[1] - 1), A), n, g),
            lambda img, i, e: img.__setitem__((i, img.shape[1] - 1, slice(None)), e),
        ],
        'T': [
            0,
            lambda img: np.vstack((np.full((1, img.shape[1], img.shape[2]), np.nan, dtype = float), img)),
            lambda img: img.shape[1],
            lambda img, w, i, A, n, g: distance(img_n_vector(img, w, (0, i), A), n, g),
            lambda img, i, e: img.__setitem__((0, i, slice(None)), e),
        ],
        'B': [
            0,
            lambda img: np.vstack((img, np.full((1, img.shape[1], img.shape[2]), np.nan, dtype = float))),
            lambda img: img.shape[1],
            lambda img, w, i, A, n, g: distance(img_n_vector(img, w, (img.shape[0] - 1, i), A), n, g),
            lambda img, i, e: img.__setitem__((img.shape[0] - 1, i, slice(None)), e),
        ],
    }
    
    # Matches pixels to sides
    if isinstance(pixels, (list, np.ndarray, tuple)):
        if len(pixels) != len(sides):
            raise ValueError('Invalid expansion')
        else:
            for i, char in enumerate(sides):
                LRTB[char][0] += pixels[i]
    else:
        LRTB[sides] = pixels
    
    # Pixel prediction on each side
    while sum(LRTB[d][0] for d in 'LRTB') > 0:
        for direction, metadata in LRTB.items():
            if metadata[0] > 0:
                metadata[0] -= 1
                img = metadata[1](img)
                for i in range(metadata[2](img)):
                    bgr = list()
                    for channel in [0, 1, 2]:
                        n_vectors = bgr_n_vectors[channel]
                        best_dist = 1e10
                        approx_neigh = dict()
                        for n_vector, UVs in n_vectors.items():
                            ranked_n_vector = list(n_vector)
                            noise_sum = 0
                            total_feature = sum(n_vector)
                            while round(noise_sum, 2) < noise_reduc:
                                noise_sum += ranked_n_vector[-1] / total_feature
                                del ranked_n_vector[-1]
                            ranked_n_vector = tuple(ranked_n_vector)
                            if len(ranked_n_vector) == 0: # For cases where noise reducation removes image
                                continue
                            for UV, As in UVs.items():
                                # Approximate best neighbors
                                # Not 100% certain, but should be faster than having to loop the entire dictionary to remove larger distances
                                for A, centers in As.items():
                                    dist = metadata[3](img[:, :, channel], width, i, np.frombuffer(A).reshape((width ** 2, len(n_vector)))[:, :len(ranked_n_vector)], ranked_n_vector, g)
                                    if dist < best_dist * (1 + error):
                                        if dist not in approx_neigh:
                                            approx_neigh[dist] = list()
                                        approx_neigh[dist].extend(centers)
                                        if dist < best_dist:
                                            best_dist = dist
                                            
                        # Best neighbors based on determined best_dist 
                        best_neighs = dict()
                        for dist, centers in approx_neigh.items():
                            if dist <= best_dist * (1 + error):
                                best_neighs[dist] = centers
                        del approx_neigh
        
                        # Histogram of centers
                        best_distances = list(best_neighs.keys())
                        st_dev = 0
                        if len(best_distances) > 2:
                            st_dev = np.std(np.array(best_distances), ddof = 1) # Standard Deviation of the Distances: https://numpy.org/doc/2.1/reference/generated/numpy.std.html
                        histogram = dict()
                        weight = 1
                        for dist, centers in best_neighs.items():
                            if not uniform and st_dev > 1e-10 and not np.isnan(st_dev):
                                # Centers weighed based on Gaussian PDF: https://www.cs.umd.edu/~djacobs/CMSC733/MarkovModelsMRFs.pdf
                                # You can try to implement a Gibbs Distribution(mentioned in the Efros & Leung Paper): #https://theory.physics.manchester.ac.uk/~judith/stat_therm/node87.html
                                weight = np.exp(-(((dist - best_dist)/ st_dev) ** 2) / 2)
                                if weight == 0:
                                    weight = 1
                            # Counter Documentation: https://docs.python.org/3/library/collections.html#collections.Counter 
                            hist = Counter(centers)
                            for center, count in hist.items():
                                if center not in histogram:
                                    histogram[center] = 0
                                histogram[center] += weight * count
                        try:
                            index = np.random.choice(len(list(histogram.keys())), p = np.array(list(histogram.values())) / sum(histogram.values()))
                            bgr.append(list(histogram.keys())[index])
                        except ValueError: # For cases where noise reducation removes image
                            bgr.append(0)
                    bgr = color_vocabulary[color_tree.query(bgr)[1]]
                    metadata[4](img, i, bgr) # Sets the pixel to the predicted value
    return img


def display_image(img, name = 'None'):
    screen_width, screen_height = 1000, 1000  # Default screen dimensions
    img = img.astype(np.uint8)  # Makes image compatible with OpenCV

    # OpenCV Documentation: https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
    cv.namedWindow(name, cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(name, screen_width, screen_height) 
    img = cv.resize(img, (screen_width, screen_height), interpolation = cv.INTER_NEAREST_EXACT)  # Resizes image
    cv.imshow(name, img)  # Shows image
    cv.waitKey(0)  # Displays image indefinitely
