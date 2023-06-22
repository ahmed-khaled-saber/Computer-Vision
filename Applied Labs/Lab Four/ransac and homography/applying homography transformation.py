import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('a.png',0)          # queryImage
img2 = cv2.imread('b.png',0)          # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints an[pllld descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
print(src_pts.shape)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
matchesMask = mask.ravel().tolist()


cv2.imshow("Img1",img1)
cv2.imshow("Img2",img2)
result = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))
cv2.imshow("Img1_wrapped", result)

cv2.waitKey(0)

