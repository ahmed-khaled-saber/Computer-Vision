import cv2
import numpy as np

def get_gradients_xy(img, ksize):
    """
    dirivatives may be negative indicate strong derivatives but in -ve Directions
    Image Values is in uint8, All Postives
    so, we need to pass cv2.CV_16S indicate we want the returned Intesities allowed to be -ve s
    and then we absolute the -ve Edges, and tranform the Image Type back to uint8
    """
    # print(img.dtype)
    # الصورة تكون من نوع 0-255 ولكن الـ سوبيل قد يحسب نتيجة سالبة كتفاضل قوي في الإتجاه السالب ولذلك كان لابد تغيير نوع الصورة الراجعة
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1,0,ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0,1,ksize=ksize)
    
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)

    sobelx = np.uint8(sobelx)
    sobely = np.uint8(sobely)

    return sobelx, sobely


def rescale(img, min,max):
    img = (img-img.min())/float(img.max()-img.min())
    img = min + img * (max-min)
    return img