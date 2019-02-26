import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image
import sys
import random
import pickle

from keypoint_detector import detect_keypoints

# match features from img1 (P1) and img2 (P2), based on descriptor size l and ratio r
def match_features(img1, P1, img2, P2, l, r):
    matches = []
    size = int((l - 1)/2)

    for idx1, p1 in enumerate(P1):
        best_err = np.Inf
        best_pair = None
        second_err = np.Inf
        second_pair = None
        for idx2, p2 in enumerate(P2):
            # for each lxl img descriptor, compare each pixel, take total squared err
            err = 0
            for i in range(-size, size+1):
                for j in range(-size, size+1):
                    pxl1 = get_pixel(img1, p1, i, j)
                    pxl2 = get_pixel(img2, p2, i, j)
                    err += (pxl1 - pxl2)**2
            # keep track of best matching err
            if (err < best_err):
                second_err = best_err
                second_pair = best_pair
                best_err = err
                best_pair = [p1, p2]
            elif (err < second_err):
                second_err = err
                second_pair = [p1, p2]
        # if difference between error is greater than ratio r, add to feature matches
        # print('best err:', best_err, second_err)
        if (best_err < second_err * r):
            matches.append(best_pair)
    return matches

# get pixel value at [x+i, y+j] or nearest pixel
def get_pixel(img, pt, i, j):
    height, width = img.shape
    x,y = pt
    i += x
    j += y
    if i < 0:
        i = 0
    elif i >= height:
        i = height-1
    if j < 0:
        j = 0
    elif j >= height:
        j = width-1
    return img[i,j]


#############################################################################
##############################       MAIN     ###############################
#############################################################################

# import image and convert to grayscale
img1 = plt.imread('class_photo1.jpg')
img1 = img1.mean(axis=2)
w1, h1 = img1.shape

img2 = plt.imread('class_photo2.jpg')
img2 = img2.mean(axis=2)
w2, h2 = img2.shape

# # get keypoints
# print('detecting keypoints...')
# P1 = detect_keypoints(img1)
# P2 = detect_keypoints(img2)
#
# # save keypoints to file
# print('keypoints detected. Saving...')
# keypoints = [P1, P2]
# keypoints_file = open('keypoints.pkl', 'wb')
# pickle.dump(keypoints, keypoints_file)
# keypoints_file.close()

# # load keypoints from file
# print('loading keypoints...')
# keypoints_file = open('keypoints.pkl', 'rb')
# keypoints = pickle.load(keypoints_file)
# P1,P2 = keypoints
#
# # print('KEYPOINTS (img1): \n', P1)
# # print('KEYPOINTS (img2): \n', P2)
#
# # descriptor size
# l = 21
# # required ratio of best-to-secondbest [suggested 0.5-0.7]
# r = 0.5
#
# # get feature matches
# print('getting matches...')
# matches = match_features(img1, P1, img2, P2, l, r)
#
# # save matches to file
# matches_file = open('matches.pkl', 'wb')
# pickle.dump(matches, matches_file)

# load matches from file
print('loading keypoints...')
matches_file = open('matches.pkl', 'rb')
matches = pickle.load(matches_file)

# print('NUM MATCHES', len(matches))
# print('MATCHES: \n', matches)

img_pair = np.column_stack((img1, img2))
plt.imshow(img_pair, cmap='gray')

# plt.scatter(0, 0, c='r')
# plt.scatter(h1, w1, c='r')
# plt.scatter(h1+h2, w2, c='r')

for match in matches:
    x1,y1 = match[0]
    x2,y2 = match[1]
    y2 += h1
    plt.scatter(y1,x1, c='r')
    plt.scatter(y2,x2, c='b')
    plt.plot([y1,y2], [x1,x2], 'g--')
plt.show()
