import numpy as np
import cv2 as cv

# Constants for color channels
BLUE, GREEN, RED = 0, 1, 2

def img_SVD(file, channel):
  u, s, v = np.linalg.svd(cv.imread(file)[:, :, channel], False)
  # We append to for time complexity compared to delete()
  U, S, V = list(), list(), list()
  for c in range(len(s)):
    if s[c] > 1e-8:
      U.append(u[:, c])
      S.append(s[c])
      V.append(v[c, :])
  return np.array(U), np.array(S), np.array(V) # Turn list into arrays
      


file = 'Patterns\\pattern_2.png'
U, S, V = img_SVD(file, RED)
print(len(U))
print(len(S))
print(len(V))
print(U.T @ np.diag(S) @ V)