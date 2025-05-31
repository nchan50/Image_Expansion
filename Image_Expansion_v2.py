import os # Manage directories and files
import pickle # Store objects in a file
from collections import Counter # Creates a dictionary from a list

import cv2 as cv # Image Comprehension
from numba import njit # Faster looping
import numpy as np # Arrays and Relevant Operations
from scipy.signal.windows import gaussian # Gaussian distributions


def img_neighborhoods(img, width):
    assert width % 2 == 1 # Restrict the width to being odd(though even widths can also work)

    neighborhoods = dict() # Neighborhood: [(b, g, r), ...]
    half = width // 2
    
    for r, c in np.ndindex(img.shape[:2]):
        if r >= half and c >= half and r < img.shape[0] - half and c < img.shape[1] - half: # Avoids neighborhoods clipping outside the edge
            neighborhood = img_neighborhood(img, width, (r, c))

            if neighborhood not in neighborhoods.keys():
                neighborhoods[neighborhood] = list()
            neighborhoods[neighborhood].append(tuple(img[r, c, :].flatten()))

    return neighborhoods


def img_neighborhood(img, width, center):
    neighborhood = np.full((width, width, 3), np.nan, dtype = float)
    half = width // 2
    r, c = center
    
    # Bounds of the patch in the image
    r_start = max(r - half, 0)
    r_end = min(r + half + 1, img.shape[0])
    c_start = max(c - half, 0)
    c_end = min(c + half + 1, img.shape[1])
    
    # Start indices in neighborhood
    r_start_neigh = half - (r - r_start)
    c_start_neigh = half - (c - c_start)

    patch = img[r_start:r_end, c_start:c_end, :]
    neighborhood[r_start_neigh : r_start_neigh + patch.shape[0], c_start_neigh : c_start_neigh + patch.shape[1]] = patch # Slices the patch across all channels

    neighborhood[half, half, :] = np.nan

    return neighborhood.tobytes() # Makes neighborhoods hashable for dictionaries


# Numba jit Documentation: https://numba.pydata.org/numba-doc/dev/user/5minguide.html 
@njit
def distance(a, b, g):
    assert len(a) == len(b)
    a = np.frombuffer(a, dtype = float).reshape(-1) # Unhashes into ndarray
    b = np.frombuffer(b, dtype = float).reshape(-1) # Unhashes into ndarray

    known = 0
    GSSD = 0
    for i in range(len(a)):
        if np.isnan(a[i]) or np.isnan(b[i]) or i % (len(a) // 3) == len(a) // 6:
            continue
        known += 1
        GSSD += g[i % (len(a) // 3)] * (a[i] - b[i]) ** 2
    return GSSD / (known / 3) if known > 0 else 1e10


def expand_image(file, width, error, pixels, sides = 'L', uniform = False, samples = None):
    # pixels: individual or list of integers of number of pixels to expand each side by
    # sides: determines which sides to extend. left(L), right(R), top(T), bottom(B)
    # noise_reduc: percentage[0, 1] of noise to remove(proxy for rank reduction)
    # uniform: determines if uniform or gaussian probability distribution is applied to the centers
    # samples: non-parametric or parametric image file(s)
    
    # Creates the gausian kernel
    # Gaussian Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html 
    gaussian_dist = gaussian(width, 1)
    g = np.outer(gaussian_dist, gaussian_dist).flatten()
    g_sum = np.sum(g)
    g *= (g_sum - g[len(g) // 2]) / g_sum # Normalizes kernel for ignoring center
    
    os.makedirs('Patterns', exist_ok = True)
    img = cv.imread(f'Patterns\\{file}').astype(float)
    if width > img.shape[0] or width > img.shape[1]:
        print(f'Width of {width} px is too large')
        return
    
    # Allows parametric image expansion 
    os.makedirs('Samples_v2', exist_ok = True)
    if samples == None:
        sample_path = f'Samples_v2\\{file}\\width_{width}.pkl'
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                neighborhoods = pickle.load(f)
        else:
            neighborhoods = img_neighborhoods(cv.imread(f'Patterns\\{file}').astype(float), width)
            os.makedirs(os.path.dirname(sample_path), exist_ok = True)
            with open(sample_path, 'wb') as f:
                pickle.dump(neighborhoods, f)
    if isinstance(samples, str):
        sample_path = f'Samples_v2\\{samples}\\width_{width}.pkl'
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                neighborhoods = pickle.load(f)
        else:
            neighborhoods = img_neighborhoods(cv.imread(f'Patterns\\{samples}').astype(float), width)
            os.makedirs(os.path.dirname(sample_path), exist_ok = True)
            with open(sample_path, 'wb') as f:
                pickle.dump(neighborhoods, f)
    if isinstance(samples, (list, np.ndarray, tuple)):
        neighborhoods = dict()
        for sample in samples:
            sample_path = f'Samples_v2\\{sample}\\width_{width}.pkl'
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    sample_neighborhoods = pickle.load(f)
            else:
                sample_neighborhoods = img_neighborhoods(cv.imread(f'Patterns\\{sample}').astype(float), width)
                os.makedirs(os.path.dirname(sample_path), exist_ok = True)
                with open(sample_path, 'wb') as f:
                    pickle.dump(sample_neighborhoods, f)
            for neighborhood, centers in sample_neighborhoods.items():
                if neighborhood not in neighborhoods.keys():
                    neighborhoods[neighborhood] = centers
                else:
                    neighborhoods[neighborhood].append(centers)
                    
    LRTB = {
        'L': [
            0,
            lambda img: np.hstack(
                (np.full((img.shape[0], 1, 3), np.nan, dtype = float), img)
            ),
            lambda img: img.shape[0],
            lambda img, w, i: img_neighborhood(img, w, (i, 0)),
            lambda img, i, e: img.__setitem__((i, 0, slice(None)), e),
        ],
        'R': [
            0,
            lambda img: np.hstack(
                (img, np.full((img.shape[0], 1, 3), np.nan, dtype = float))
            ),
            lambda img: img.shape[0],
            lambda img, w, i: img_neighborhood(img, w, (i, img.shape[1] - 1)),
            lambda img, i, e: img.__setitem__((i, img.shape[1] - 1, slice(None)), e),
        ],
        'T': [
            0,
            lambda img: np.vstack(
                (np.full((1, img.shape[1], 3), np.nan, dtype = float), img)
            ),
            lambda img: img.shape[1],
            lambda img, w, i: img_neighborhood(img, w, (0, i)),
            lambda img, i, e: img.__setitem__((0, i, slice(None)), e),
        ],
        'B': [
            0,
            lambda img: np.vstack(
                (img, np.full((1, img.shape[1], 3), np.nan, dtype = float))
            ),
            lambda img: img.shape[1],
            lambda img, w, i: img_neighborhood(img, w, (img.shape[0] - 1, i)),
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
                metadata[0] -= 1 # Iterate over pixels on each side
                img = metadata[1](img) # Adds a row/column of np.nan to the respective side
                for i in range(metadata[2](img)): # Iterate over the side
                    neighborhood = metadata[3](img, width, i) # Neighborhood of the unknown pixel
                    best_dist = 1e10
                    
                    # Approximate neighbors based on a decreasing best_dist 
                    approx_neigh = dict()
                    for n in neighborhoods:
                        dist = distance(n, neighborhood, g)
                        if dist < best_dist * (1 + error):
                            approx_neigh[n] = dist
                            if dist < best_dist:
                                best_dist = dist
                    # Best neighbors based on determined best_dist 
                    best_neigh = dict()
                    for neigh, dist in approx_neigh.items():
                        if dist <= best_dist * (1 + error):
                            best_neigh[neigh] = dist
                    del approx_neigh
                    
                    # Histogram of centers
                    best_neigh_values = list(best_neigh.values())
                    if len(best_neigh_values) > 2:
                        st_dev = np.std(np.array(best_neigh_values), ddof = 1) # Standard Deviation of the Distances: https://numpy.org/doc/2.1/reference/generated/numpy.std.html
                    else:
                        st_dev = 0
                    
                    histogram = dict()
                    weight = 1
                    for neigh in best_neigh:
                        if not uniform and st_dev > 1e-10 and not np.isnan(st_dev):
                            weight = np.exp(-((best_neigh[neigh] / st_dev) ** 2) / 2) # Centers weighed based on Gaussian PDF: https://www.cs.umd.edu/~djacobs/CMSC733/MarkovModelsMRFs.pdf 
                            if weight == 0:
                                weight = 1
                        # Counter Documentation: https://docs.python.org/3/library/collections.html#collections.Counter 
                        hist = Counter(neighborhoods[neigh])
                        for center, count in hist.items():
                            if center not in histogram:
                                histogram[center] = 0
                            histogram[center] += weight * count
                    index = np.random.choice(len(list(histogram.keys())), p = np.array(list(histogram.values())) / sum(histogram.values()))
                    metadata[4](img, i, list(histogram.keys())[index]) # Sets the pixel to the predicted value
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