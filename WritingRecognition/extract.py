# Sample code for AI Coursework
# Extracts the letters from a printed page.

# Keep in mind that after you extract each letter, you have to normalise the size.
# You can do that by using scipy.imresize. It is a good idea to train your classifiers
# using a constast size (for example 20x20 pixels)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize, imsave
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.morphology import label
import cv2
import time
import pytesseract
from pytesseract import Output
from skimage.measure import regionprops

class Extract_Letters:
    def extractFile(self, filename):

        image = cv2.imread(filename,1)
        print(image.shape)
        copy = image.copy()
        im2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        bw = (image < 120).astype(np.float)
        #im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        #im2 = cv2.GaussianBlur(im2, (5, 5), 0)
        #im2 = cv2.medianBlur(im2, 3)
        ret,thresh = cv2.threshold(im2,127,255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((10, 1), np.uint8)
        #kernel = np.ones((5,5),np.uint8)
        #erosion = cv2.erode(im2,kernel,iterations = 1)
        #gradient = cv2.morphologyEx(im2, cv2.MORPH_GRADIENT, kernel)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(img_dilation.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        order = []
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        #fig, ax = plt.subplots(1, 1, figsize=(18, 32))
        #n_char = np.shape(ctrs)[0]
        for i, ctr in enumerate(ctrs):
            x,y,w,h = cv2.boundingRect(ctr)
            minr = y
            minc = x
            maxr = y + h
            maxc = x + w
            region = (minr, minc, maxr, maxc)
            if h > 15 and h < 60:
                letter = image[y:y + h, x:x + w]
                #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                rect = cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),1)
                order.append(region)

        
        # sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        for i in range(len(order)):
            print("LINE {}: {}".format(i,order[i]))
            print("")

        # worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])

        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                print("ABS VALUE ({}) < ABS VALUE ({})".format((abs(character[0] - first_in_line[0])),((first_in_line[2] - first_in_line[0])) ))
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                print("ABS VALUE ({}) < ABS VALUE ({})".format((abs(character[0] - first_in_line[0])),((first_in_line[2] - first_in_line[0])) ))
                first_in_line = character
                counter += 1
                lines[counter].append(character)
        
        #row = order[1][1]
        #for character in order:
            #if first_in_line == '':
                #first_in_line = character
                #lines[counter].append(character)
            #elif character[1] <= row:
                #lines[counter].append(character)
            #else:
             #   first_in_line = character
              #  counter += 1
               # row = character[1]
                #lines[counter].append(character)
        
        for x in range(len(lines)):
            lines[x].sort(key=lambda tup: tup[1])

        lines.reverse()

        for i in range(len(lines)):
            print("LINE {}: {}".format(i,lines[i]))
            print("")
        

        final = list()
        prev_tr = 0
        prev_line_br = 0

        for i in range(len(lines)):
            #print("i: {}".format(i))
            #print("LINE {}: {}".format(i, lines[i]))
            for j in range(len(lines[i])):
                #print("lines[{}]: ".format(j))
                tl_2 = lines[i][j][1]
                bl_2 = lines[i][j][0]
                tl, tr, bl, br = lines[i][j]
                #print("tl, tr, bl, br: {}, {}, {}, {}".format(tl, tr, bl, br))
                #print('tl_2: {}, bl_2: {}'.format(tl_2, bl_2))
                #print('prev_tr: {}, prev_line_br: {}'.format(prev_tr, prev_line_br))
                if tl_2 >= prev_tr and bl_2 >= prev_line_br:
                    tl, tr, bl, br = lines[i][j]
                    letter_raw = bw[tl:bl, tr:br]
                    #print("tl, tr, bl, br: {}, {}, {}, {}".format(tl, tr, bl, br))
                    letter_norm = resize(letter_raw, (20, 20))
                    final.append(letter_raw)
                    #print("LETTER RAW: {}".format(bw[tl:bl, tr:br]))
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i]) - 1):
                    prev_line_br = lines[i][j][2]
            prev_tr = 0
            tl_2 = 0

        print ('Characters recognized: ' + str(len(final)))
        return final

    def __init__(self):
        print("Extracting characters...")

start_time = time.time()
extract = Extract_Letters()
training_files = ['./ocr/training/lowercase.png']

folder_string = 'abcdefghijklmnopqrstuvwxyz'
name_counter = 600
for files in training_files:
    letters = extract.extractFile(files)
    string_counter = 0

    for i in letters:
        if string_counter > 25:
            string_counter = 0
        imsave('./training_type/' + str(folder_string[string_counter]) + '/' + str(name_counter) + '_snippet.png', i)

        string_counter += 1
        name_counter += 1

print (time.time() - start_time, "seconds" )
