import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# ===============================================================================
def Apply_SIFT(img_a,img_b):
    s = cv2.SIFT_create()
    kpoints1, descriptor1= s.detectAndCompute(img_a,None)
    kpoints2, descriptor2= s.detectAndCompute(img_b,None)
    
    m = cv2.BFMatcher()
    matches = m.match(descriptor1,descriptor2)
    matches = sorted(matches,key= lambda x:x.distance)
    
    return matches, kpoints1,descriptor1,kpoints2,descriptor2
# ===============================================================================
# ===============================================================================
def Apply_David_Lowe_Ratio(descp1,descp2, RATIO):
    m = cv2.BFMatcher()
    matches = m.knnMatch(descp1,descp2,2)
    
    accpted_matches = []
    for m in matches:
        if((m[0].distance/m[1].distance) < RATIO):
            accpted_matches.append(m[0])
    return accpted_matches
# ===============================================================================
# ===============================================================================
def Apply_CrossCheck(descp1,descp2):
    m = cv2.BFMatcher()
    match1 = m.match(descp1, descp2)
    match2 = m.match(descp2, descp1)
    
    acepted_matches = []
    for m1 in match1:
        for m2 in match2:
            if m2.queryIdx == m1.trainIdx and m1.queryIdx == m2.trainIdx:
                acepted_matches.append(m1)
    return acepted_matches
# ===============================================================================
# ===============================================================================
def Judge_Similarity(filtered_matches, kypts1, kypts2, THRESHOLD):
    similarity_score = len(filtered_matches)/ min(len(kypts1),len(kypts2))
    if similarity_score < THRESHOLD:
        return f"This Pair is not SIMILAR!, because {similarity_score} is greater than our t {THRESHOLD}", similarity_score
    else:
        return f"This Pair is SIMILAR!, because {similarity_score} is less than our t {THRESHOLD}", similarity_score
# ===============================================================================
# ===============================================================================
def Apply_Draw_matches(img_a, kypts_a, img_b, kypts_b, filtered_matches, similarity_score , bool):
    drawing = cv2.drawMatches(img_a, kypts_a, img_b, kypts_b, filtered_matches, flags=2, outImg=None)
    plt.imshow(drawing), plt.show()
# ===============================================================================
# ===============================================================================
def Apply_Homography(filtered_matches, kypts1, kypts2, THRESHOLD):
    src_pts = np.float32([kypts1[i.queryIdx].pt for i in filtered_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kypts2[i.trainIdx].pt for i in filtered_matches]).reshape(-1, 1, 2)
    (H, status) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, THRESHOLD)
    return (H, status)
# ===============================================================================
# ===============================================================================
def Apply_Refined_RANSAC_Draw_matches(img_a, kypts1, img_b, kypts2, filtered_matches, status):
    status = status.ravel().tolist()
    # Mask Matches after RANSAC
    drawing = cv2.drawMatches(img_a, kypts1, img_b, kypts2, filtered_matches, None, matchColor=(0, 255, 0), matchesMask=status, flags=2)
    plt.imshow(drawing), plt.show()
# ===============================================================================



""" Relative Path only works taht way, i wish if YOU Change it to fit the Host """
images = os.listdir('Lab Four/Assignment2/assignment data')
path   = 'Lab Four/Assignment2/assignment data'
for i in images :
    query_path = os.path.join(path,i)
    query_img = cv2.imread(path,0)
    for j in images:
        if(i==j): continue;
        elif(i[5]==j[5]):    # Example:: image1a.jpeg image1b.jpeg
            train_path = os.path.join(path, j)
            train_img = cv2.imread(train_path, 0)
            # print(i , j)
            
            # # Apply SIFT, get Matches
            matches, kypts1, descriptor1, kypts2, descriptor2 = Apply_SIFT(query_img,train_img)
            
            # CHECKPOINT 
            print(matches[0])

            ### Filtering by David Lowe’s Ratio
            ratio_matches = Apply_David_Lowe_Ratio(descriptor1,descriptor2,ratio=0.7)

            ### Filtering by Cross Check
            crosscheck_matches = Apply_CrossCheck(descriptor1, descriptor2)
            
            # get the most Candidates Matches.
            filtered_matches = []
            for m1 in ratio_matches:
                for m2 in crosscheck_matches:
                    if m1.queryIdx == m2.queryIdx and m1.trainIdx == m2.trainIdx:
                        filtered_matches.append(m1)
                        
            ### Judge_Similarity, 
            THRESHOLD = 0.01
            similarity_msg, similarity_score = Judge_Similarity(filtered_matches, kypts1, kypts2, THRESHOLD)
            print(similarity_msg)    # similarity Message

            ### Draw matches after compressing matches
            Apply_Draw_matches(query_img, kypts1, train_img, kypts2, filtered_matches, similarity_score)
            
            ### Homography
            (H, status)= Apply_Homography(filtered_matches, kypts1, kypts2,THRESHOLD=4.0)
            Apply_Refined_RANSAC_Draw_matches(query_img, kypts1, train_img, kypts2, filtered_matches, status)

