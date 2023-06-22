# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
import glob

from panorama import PanoramaGenerator
import cv2


images = glob.glob("images/*.png")

left_images = [img for img in images if img.__contains__("_left_")]
left_images.sort()
right_images = [img for img in images if img.__contains__("_right_")]
right_images.sort()

for left_img_path, right_img_path in zip(left_images, right_images):
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    imageA = cv2.imread(right_img_path)
    imageB = cv2.imread(left_img_path)
    imageA = cv2.resize(imageA,(400,200))
    imageB = cv2.resize(imageB,(400,200))

    # stitch the images together to create a panorama
    stitcher = PanoramaGenerator()
    (result, vis) = stitcher.generate_panorama_image([imageA, imageB], showMatches=True)

    # show the images
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)