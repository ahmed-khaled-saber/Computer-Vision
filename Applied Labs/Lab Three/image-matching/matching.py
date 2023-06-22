import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('Lab Three/image-matching/box.png',0)          # queryImage
img2 = cv2.imread('Lab Three/image-matching/box_in_scene.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints an[pllld descriptors with SIFT
kp1, des1 = sift.detectAndCompute(cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),None)
kp2, des2 = sift.detectAndCompute(cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)
#matches = bf.knnMatch(des1,des2,2)
i=0
for match in matches:
    print(match.queryIdx)
    print(match.trainIdx)
    print(match.distance)
    print(match.imgIdx)
    print("------------------------------------------------------------------------------------")

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,
                       img2,kp2,
                       matches[0:20],
                       flags=2, outImg=None)

plt.imshow(img3),plt.show()