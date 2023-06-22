from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

def matchImages():
    query_image = cv2.imread('Lab Three/hands on/query.jpg',0)          # queryImage
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(cv2.normalize(query_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),None)
    img_names = os.listdir('tiny_data')
    bf = cv2.BFMatcher()
    
    query_to_train_distances = []
    sum = 0
    for i in img_names:
        train_img_path = os.path.join('tiny_data/',i)
        img = cv2.imread(train_img_path, 0)
        kp2, des2 = sift.detectAndCompute(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),None)
        matches=bf.match(des1,des2)
        matches = sorted(matches, key=lambda x: x.distance)
        for m in matches[0: 10]:
            sum += m.distance
        query_to_train_distances.append(sum)
        sum=0

    for i in range(len(img_names)):
        train_img = cv2.imread(os.path.join('tiny_data/',img_names[i]),0)
        s = "Distance between current image and query image is: "+str(query_to_train_distances[i])
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(1, 2, 1)
        plt.imshow(train_img, cmap='gray')
        plt.axis('off')

        ax = plt.subplot(1, 2, 2)
        plt.imshow(query_image, cmap='gray')
        plt.axis('off')
        plt.text(-250, -10, s, fontsize=24)
        plt.show()

    index_min_distance = query_to_train_distances.index(min(query_to_train_distances))
    print("minimum distance is: ")
    print(min(query_to_train_distances))
    print("best image Name is:")
    print(img_names[index_min_distance])

matchImages()