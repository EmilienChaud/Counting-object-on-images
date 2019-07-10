import numpy as np
import scipy
from skimage.feature import blob_log
import scipy
import joblib
import cv2

filename = '41.jpg'

def load_image(path_data, filename):
    
    image_dotted = cv2.imread(path_data +"/TrainDotted/" + filename)
    image_normal = cv2.imread(path_data + "/Train/" + filename)
    
    return image_dotted, image_normal

def get_point_position(image_dotted, image_normal):
    
    #image_1 = cv2.imread(path_data +"/TrainDotted/" + filename)
    #image_2 = cv2.imread(path_data + "/Train/" + filename)
    img1 = cv2.GaussianBlur(image_dotted,(5,5),0)

    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_dotted,image_normal)
    mask_1 = cv2.cvtColor(image_dotted, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4,axis=2)

    # detect blobs
    blobs = blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1,
                     threshold=0.05)
    
    return blobs



def give_blobs_color(image_dotted, blobs):
    
    color_blobs = [] 

    for blob in blobs:
        color_blobs.append(list(image_dotted[int(blob[0]), int(blob[1]), :]))
    
    return np.array(color_blobs)
    

kmeans = joblib.load('Model_param/Lazy_Kmean')

def give_color_kind(color):
    

    
    return kmeans.predict(color)





def make_a_disk(image_shape, x, y, r):
    
    xx, yy = np.mgrid[:image_shape[0], :image_shape[1]]
    # circles contains the squared distance to the (100, 100) point
    # we are just using the circle equation learnt at school
    circle = (xx - x) ** 2 + (yy - y) ** 2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    disk = np.logical_and(circle < r^2, circle >=0)
    
    return 1*disk

    
    





