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

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
print(src_pts.shape)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
matchesMask = mask.ravel().tolist()



# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,
                       img2,kp2,
                       matches,
                       flags=2, outImg=None)

img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,
                       matchColor=(0,255,0),# draw matches in green color
                       matchesMask=matchesMask,# draw only inliers
                       flags=2)

plt.imshow(img3, 'gray')
plt.show()
plt.imshow(img4, 'gray')
plt.show()