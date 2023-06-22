import cv2
import numpy as np
import cv2 as cv

image = cv.imread('/home/ahmed/FCIS-Seniority/Computer Vision/PROJECT/SeaLifeClassificationDetection/test/FishDataset662_png.rf.NEl3f23MIAvV4j8HDfga.jpg')
reshaped = image.reshape((-1, 3))

# make sure its float32 data type
reshaped = np.float32(reshaped)

# define my criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply k-means clustering
ret, label, center = cv.kmeans(reshaped, 2, None, criteria, 10, cv.KMEANS_PP_CENTERS)
# Apply mean shift
bandwidth = 20
ms = cv2.pyrMeanShiftFiltering(image, sp=bandwidth, sr=bandwidth)
ms_labels = cv2.cvtColor(ms, cv2.COLOR_RGB2GRAY)
ms_segmented = ms_labels.reshape(image.shape[:2])

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((image.shape))

cv2.imwrite('segmented.jpg', res2)
cv2.imshow('Image', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('segmented.jpg', ms_segmented)
cv2.imshow('Image', ms_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()