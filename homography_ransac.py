import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image
import sys
from random import shuffle
import pickle
import skimage.transform as skt

# local imports
from feature_matching import match_features, plot_and_savefig_matches
from keypoint_detector import detect_keypoints, plot_and_savefig_keypoints

# Finds a Homography from points in img_1 to corresponding points in img_2
def Homography(matches):
    A = np.zeros((8,9))
    pts_1 = [x[0] for x in matches]
    pts_2 = [x[1] for x in matches]
    for i in range(4):
        u1,v1,_ = pts_1[i]
        u2,v2,_ = pts_2[i]
        A[(2*i):(2*(i+1))] = [[ 0,  0, 0, -u1, -v1, -1,  v2*u1,  v2*v1,  v2],
                              [u1, v1, 1,   0,   0,  0, -u2*u1, -u2*v1, -u2]]
    U,Sigma,Vt = np.linalg.svd(A)
    h = Vt[Vt.shape[0]-1]
    h = h.reshape((3,3))
    return h


def RANSAC(in_matches,number_of_iterations=3,n=4,r=4,d=10):

    matches = in_matches.copy()
    H_best = np.array([[1,0,0],[0,1,0],[0,0,1]])
    inliers_best = []
    residual_best = np.Inf
    num_inliers_best = 0
    num_matches = len(matches)
    r = r**2

    # convert (u,v) to (u,v,1)
    for i in range(num_matches):
        match = matches[i]
        if len(match[0]) == 2:
            match[0].append(1)
        if len(match[1]) == 2:
            match[1].append(1)
    print(matches)

    for i in range(number_of_iterations):
        print('Iteration:', i)
        # 1. Select a random sample of length n from the matches
        shuffle(matches)
        sample_matches = matches[0:n+1]
        pts_1 = [x[0] for x in matches]
        pts_2 = [x[1] for x in matches]
        pts_pred = [0 for x in matches]

        # 2. Compute a homography based on these points using the methods given above
        print('calculating homography...')
        H = Homography(sample_matches)
        print('homography calced...')

        # 3. Apply this homography to the remaining points that were not randomly selected
        for i in range(num_matches):
            # print(H, pts_1[i])
            x,y,w = np.matmul(H,pts_1[i])
            pts_pred[i] = [x/w, y/w]

        residual_total = 0
        num_inliers = 0
        inliers_list = []
        for i in range(num_matches):
            # 4. Compute the residual between observed and predicted feature locations
            u1,v1,_ = pts_2[i]
            u2,v2 = pts_pred[i]
            residual = (u1-u2)**2 + (v1-v2)**2
            residual_total += residual
            # 5. Flag predictions that lie within a predefined distance r (3-5 pixels) from observations as inliers
            if residual > r:
                num_inliers += 1
                inliers_list.append(matches[i])

        print('number of inliers:', num_inliers)
        # 6. If number of inliers is greater than the previous best
        #    and greater than a minimum number of inliers d,
        if num_inliers > num_inliers_best:
            #    7. update H_best
            H = H.reshape((3,3))
            H_best = H
            #    8. update list_of_inliers
            inliers_best = inliers_list
            num_inliers_best = num_inliers

    return H_best, inliers_best

#############################################################################
##############################       MAIN     ###############################
#############################################################################

# import image and convert to grayscale
img1_filename = 'class_photo1.jpg'
img1_name = img1_filename.split(".")[0]
img1 = plt.imread(img1_filename)
img1 = img1.mean(axis=2)
w1, h1 = img1.shape

img2_filename = 'class_photo1.jpg'
img2_name = img2_filename.split(".")[0]
img2 = plt.imread(img2_filename)
img2 = img2.mean(axis=2)
w2, h2 = img2.shape

# output files
keypoints_filename = "keypoints.{}.{}.pkl".format(img1_name, img2_name)
keypoints1_img = "keypoints.{}.jpg".format(img1_name)
keypoints2_img = "keypoints.{}.jpg".format(img2_name)
matches_filename = "matches.{}.{}.pkl".format(img1_name, img2_name)
matches_img = "matches.{}.{}.jpg".format(img1_name, img2_name)

# Plot images together
img_pair = np.column_stack((img1, img2))
plt.imshow(img_pair, cmap='gray')
plt.show()

# # get keypoints
# print('detecting keypoints...')
# P1 = detect_keypoints(img1)
# P2 = detect_keypoints(img2)
#
# # save keypoints to file
# print('keypoints detected. Saving...')
# keypoints = [P1, P2]
# keypoints_file = open(keypoints_filename, 'wb')
# pickle.dump(keypoints, keypoints_file)
# keypoints_file.close()

# load keypoints from file
print('loading keypoints...')
keypoints_file = open(keypoints_filename, 'rb')
keypoints = pickle.load(keypoints_file)
P1,P2 = keypoints

# plot and save keypoints
plot_and_savefig_keypoints(img1,P1,keypoints1_img)
plot_and_savefig_keypoints(img2,P2,keypoints2_img)

# descriptor size
l = 21
# required ratio of best-to-secondbest [suggested 0.5-0.7]
r = 0.5

# get feature matches
print('getting matches...')
matches = match_features(img1, P1, img2, P2, l, r)

# save matches to file
print('matches found. Saving...')
matches_file = open(matches_filename, 'wb')
pickle.dump(matches, matches_file)

# load matches from file
print('loading matches...')
matches_file = open(matches_filename, 'rb')
matches = pickle.load(matches_file)
print('matches loaded.')

# plot and savefig matches
plot_and_savefig_matches(img1, img2, matches, matches_img)

# run RANSAC to find outliers
print('running RANSAC...')
print('matches:', len(matches))
H_best, list_of_inliers = RANSAC(matches)
print('inliers:', len(list_of_inliers))
print('RANSAC Results:\t H_best:{}\n inliers:\n{}'.format(H_best, list_of_inliers))

# After RANSAC, transform img2 via the transform

# H_best = np.eye(3) # CHANGE ME!
# Create a projective transform based on the homography matrix $H$
proj_trans = skt.ProjectiveTransform(H_best)
# Warp the image into image 1's coordinate system
img2_trans = skt.warp(img2,proj_trans)

# Output resulting image
plt.imshow(img2_trans, cmap='gray')
plt.show()
