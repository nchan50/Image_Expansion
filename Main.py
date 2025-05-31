import os # Manage directories and files
import time # Tracks current time

import cv2 as cv # Image Compression
import numpy as np # Arrays and Relevant Operations

import Image_Expansion_v1 as IE1
import Image_Expansion_v2 as IE2
import Image_Expansion_v3 as IE3

# Compresses Images: https://image.pi7.org/reduce-image-resolution 
# Create 10 x 10 Pixel Patterns: https://www.pixilart.com/draw

pixels = [10, 10, 10, 10]
sides = 'LRTB'
error = 0.1
time_data = str()
for i in range(1, 6):
    file = f'pattern_{i}.png'
    results_path = f'Results\\pattern_{i}.png'
    os.makedirs(results_path, exist_ok = True)
    start = time.time()
    i1 = cv.merge(IE1.expand_image(file, pixels, sides)).astype(np.uint8)
    end = time.time()
    i1_name = f'V1_{pixels}_{sides}'
    time_data += i1_name + f': {end - start}\n'
    cv.imwrite(f'{results_path}\\{i1_name}.png', i1)
    for j in range(1, 4):
        width = 1 + 2 * j
        start = time.time()
        i2 = IE2.expand_image(file, width, error, pixels, sides).astype(np.uint8)
        end = time.time()
        i2_name = f'V2_{width}_{error}_{pixels}_{sides}'
        time_data += i2_name + f': {end - start}\n'
        cv.imwrite(f'{results_path}\\{i2_name}.png', i2)
        for k in range(5):
            noise_reduc = k * 0.25
            start = time.time()
            i3 = IE3.expand_image(file, width, error, pixels, sides, noise_reduc).astype(np.uint8)
            end = time.time()
            i3_name = f'V3_{width}_{error}_{pixels}_{sides}_{noise_reduc}'
            time_data += i3_name + f': {end - start}\n'
            cv.imwrite(f'{results_path}\\{i3_name}.png', i3)
open("Results\\time_data1.txt", "x")
with open('Results\\time_data1.txt', 'a') as f:
        f.write(time_data)