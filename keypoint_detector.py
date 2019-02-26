import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import random

# local imports
from convolution import *

def detect_keypoints(g):

    h_sobelu = sobel_u_kernel()
    h_sobelv = sobel_v_kernel()
    h_gaussian = gaussian_kernel(2, 2)

    print('kernels calculated...')

    g_u = convolve(g, h_sobelu)
    g_u2 = np.multiply(g_u, g_u)
    g_u2 = convolve(g_u2, h_gaussian)

    g_v = convolve(g, h_sobelv)
    g_v2 = np.multiply(g_v, g_v)
    g_v2 = convolve(g_v2, h_gaussian)

    g_uv = np.multiply(g_u, g_v)
    g_uv = convolve(g_uv, h_gaussian)

    # Harris Response matrix
    H = np.zeros_like(g)
    # Determinant(g)
    H_det = np.multiply(g_u2, g_v2) - np.multiply(g_uv, g_uv)
    # Trace(g) (small number to prevent division by zero)
    H_tr = g_u2 + g_v2 + 1e-10
    # H = det(g)/tr(g)
    H = np.divide(H_det, H_tr)

    print('harris matrix constructed...')

    # find local maxima
    size = 1
    max = local_maxima(H, size)

    print('local maximas found...')

    ### Non-maximal Suppression
    # sort local maxima by Harris score
    random.shuffle(max)
    max = max[0:1000]
    max.sort()
    suppressedMax = []

    # the nearest local maximal neighbor of each pt with greater Harris score
    for i in range(len(max)):
        pt_curr = max[i]
        pt_near = None
        pt_dist = None
        for j in range(i+1,len(max)):
            # for each greater Harris score, calculate distance
            dist = (pt_curr[1][0] - max[j][1][0])**2 + (pt_curr[1][1] - max[j][1][1])**2
            if (pt_dist == None or pt_dist > dist):
                pt_near = max[j]
                pt_dist = dist
        if (pt_near != None):
            suppressedMax.append([pt_dist, pt_curr])

    # Sort Non-Maximal Suppressed List by distance
    suppressedMax.sort(reverse=True)
    topMax = suppressedMax[0:100]
    # get only points
    topPoints = [x[1][1] for x in topMax]

    return topPoints

# find local maxima of Harris Response matrix
def local_maxima(H, size):
    height, width = H.shape
    max = []
    # iterate over pixels
    for u in range(height):
        for v in range(width):
            # add to list if local maxima
            if check_if_local_maxima(H, size, width, height, u, v):
                max.append([H[u,v], [u,v]])

    return max

# Check if a given pixel is a local maxima
def check_if_local_maxima(H, size, width, height, u, v):
    # test pixel Harris score
    px = H[u,v]
    # iterate over neighborhood
    for j in range(u-size, u+size+1):
        for k in range(v-size, v+size+1):
            if j<0 or k<0 or j>height-1 or k>width-1:
                continue
            # skip itself
            if j==u and k==v:
                continue
            # if pixel of greater score found, not maxima
            elif px < H[j,k]:
                return False
    # else, is maxima
    return True

#############################################################################
##############################       MAIN     ###############################
#############################################################################

# # import image and convert to grayscale
# I = plt.imread('noisy_big_chief.jpeg')
# I = I.mean(axis=2)
#
# keypoints = detect_keypoints(I)
# print('KEYPOINTS: \n', keypoints)
#
# for i in range(len(keypoints)):
#     plt.scatter(x=keypoints[i][1], y=keypoints[i][0], c='b')
# plt.imshow(I,cmap=plt.cm.gray)
# plt.show()
