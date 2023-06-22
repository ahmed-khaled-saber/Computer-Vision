# # import the necessary packages
# import numpy as np
# import cv2

# class PanoramaGenerator:
#     def generate_panorama_image(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
#         (imageA, imageB) = images
#         (kpsA, featuresA) = self.detectAndDescribe(imageA)
#         (kpsB, featuresB) = self.detectAndDescribe(imageB)

#         # match features between the two images
#         M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

#         # if the match is None, then there aren't enough matched
#         # keypoints to create a panorama
#         if M is None:
#             return None

#         # otherwise, apply a perspective warp to stitch the images
#         # together
#         (matches, H, status) = M

#         # Student Code: apply the warping on the 2nd image
#         #  cv2.warpPerspective(right image, homography_matrix, (width, height))
#         # 1 line of code
#         # End Student Code
#         cv2.imshow("warp_Result", result)
#         result[0:imageB.shape[0], 0:imageB.shape[1]] = ??

#         # check to see if the keypoint matches should be visualized
#         if showMatches:
#             vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
#                 status)

#             # return a tuple of the stitched image and the
#             # visualization
#             return (result, vis)

#         # return the stitched image
#         return result

#     def detectAndDescribe(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         descriptor = cv2.xfeatures2d.SIFT_create()
#         (kps, features) = descriptor.detectAndCompute(gray, None)

#         # convert the keypoints from KeyPoint objects to NumPy arrays
#         kps = np.float32([kp.pt for kp in kps])

#         # return a tuple of keypoints and features
#         return (kps, features)

#     def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
#         # compute the raw matches and initialize the list of actual matches
#         matcher = cv2.BFMatcher()
#         rawMatches = matcher.knnMatch(featuresA, featuresB, 2) 

#         matches = []

#         # loop over the raw matches
#         for m in rawMatches:
#             # Student Code: ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
#             # That is if the distance of the top match is less than a ratio multiplied by the 2nd top match
#             # then add the top match features indices (a tuple of train and query indices) to the matches list,
#             # otherwise, do nothing. Hint: you will need DMatch.distance, DMatch.queryIdx, DMatch.trainIdx
#             # 2 lines of code
#             # End Student Code

#         # computing a homography requires at least 4 matches
#         if len(matches) > 4:
#             # Student Code: construct the two sets of points
#             # Replace the ??? with the exact words to get the keypoint of image a and image b
#             # ptsA = np.float32([??? for (_, ind) in ???])
#             # ptsB = np.float32([??? for (ind, _) in ???])
#             # Hint:
#             # An example of using list comprehension to get a list of the first element in a 2d array:
#             # first_column:  [row[0] for row in two_dimension_array]
#             # 2 lines of code:
#             # End Student Code

#             # compute the homography between the two sets of points
#             (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
#                 reprojThresh)

#             # return the matches along with the homograpy matrix
#             # and status of each matched point
#             return (matches, H, status)

#         # otherwise, no homograpy could be computed
#         return None

#     def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
#         # initialize the output visualization image
#         (hA, wA) = imageA.shape[:2]
#         (hB, wB) = imageB.shape[:2]
#         vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
#         vis[0:hA, 0:wA] = imageA
#         vis[0:hB, wA:] = imageB

#         # loop over the matches
#         for ((trainIdx, queryIdx), s) in zip(matches, status):
#             # only process the match if the keypoint was successfully
#             # matched
#             if s == 1:
#                 # draw the match
#                 ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
#                 ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
#                 cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

#         # return the visualization
#         return vis