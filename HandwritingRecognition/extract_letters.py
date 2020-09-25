# Sample code for AI Coursework
# Extracts the letters from a printed page.

# Keep in mind that after you extract each letter, you have to normalise the size.
# You can do that by using scipy.imresize. It is a good idea to train your classifiers
# using a constast size (for example 20x20 pixels)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize


image = imread('./ocr/testing/alchemist2.png',1)

#apply threshold in order to make the image binary
bw = image < 120

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared,neighbors=8)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1


fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(bw, cmap='jet')

order = []

for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    if maxr - minr > len(image) / 150:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        order.append(region.bbox)

print("Characters Recognized: {}".format(len(order)))

lines = list()
first_in_line = ''
counter = 0

# worst case scenario there can be 1 character per line
for x in range(len(order)):
    lines.append([])

for character in order:
    if first_in_line == '':
        first_in_line = character
        lines[counter].append(character)
    elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
        lines[counter].append(character)
    elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
        first_in_line = character
        counter += 1
        lines[counter].append(character)

for x in range(len(lines)):
    lines[x].sort(key=lambda tup: tup[1])

final = list()
prev_tr = 0
prev_line_br = 0

for i in range(len(lines)):
    for j in range(len(lines[i])):
        tl_2 = lines[i][j][1]
        bl_2 = lines[i][j][0]
        if tl_2 > prev_tr and bl_2 > prev_line_br:
            tl, tr, bl, br = lines[i][j]
            letter_raw = bw[tl:bl, tr:br]
            letter_norm = resize(letter_raw, (20, 20))
            final.append(letter_norm)
            prev_tr = lines[i][j][3]
        if j == (len(lines[i]) - 1):
            prev_line_br = lines[i][j][2]
    prev_tr = 0
    tl_2 = 0

print ('Characters recognized: ' + str(len(final)))