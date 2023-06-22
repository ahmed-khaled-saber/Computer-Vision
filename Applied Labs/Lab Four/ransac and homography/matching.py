import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints an[pllld descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)
#matches = bf.knnMatch(des1,des2,2)
i=0


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,
                       img2,kp2,
                       matches,
                       flags=2, outImg=None)

plt.imshow(img3),plt.show()