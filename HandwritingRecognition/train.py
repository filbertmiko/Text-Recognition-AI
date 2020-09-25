import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from letters_extractor import Extract_Letters

start_time = time.time()
extract = Extract_Letters()
training_files = ['./ocr/training/miko_dataset.png', './ocr/training/velia_dataset.png','./ocr/training/caroline_dataset.png',
                  './ocr/training/edrik_dataset.png']

folder_string = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz123456789'
name_counter = 600
for files in training_files:
    letters = extract.extractFile(files)
    string_counter = 0

    for i in letters:
        if string_counter > 60:
            string_counter = 0
        imsave('./training_type/' + str(folder_string[string_counter]) + '/' + str(name_counter) + '_snippet.png', i)

        string_counter += 1
        name_counter += 1

print (time.time() - start_time, "seconds" )
