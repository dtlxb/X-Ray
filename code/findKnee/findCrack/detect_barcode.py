# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import cv2
import numpy as np  
from matplotlib import pyplot as plt 

#img = cv2.imread("./test3.jpg")

for i in range(1, 26):
    
    img_name = './img/test'+str(i)+'.jpg'
    img = cv2.imread(img_name)

    w = img.shape[0]
    h = img.shape[1]
    img = img[int(0.1*w):int(0.9*w),:]


    img = cv2.GaussianBlur(img,(3,3),0)
    
    img = cv2.Canny(img, 50, 150) # 50， 150


    img = cv2.dilate(img, None, iterations = 2)
    #img = cv2.erode(img, None, iterations = 1)


    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,50)) #MORPH_RECT


    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)

    gray_temp = img.copy() #copy the gray image because function#findContours will change the imput image into another  
    thresh, contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #show the contours of the imput image
    #cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    plt.figure('original image with contours')
    plt.imshow(img, cmap = 'gray')
    #find the max area of all the contours and fill it with 0
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    #cv2.fillConvexPoly(img, contours[max_idx], 0)
    #show image without max connect components 
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
    out_name = './rst/rst'+str(i)+'.png'
    cv2.imwrite(out_name, img)

#plt.figure('remove max connect com'), plt.imshow(img, cmap = 'gray')
#plt.show()



#cv2.imshow('Canny', img)
#plt.imshow(imgplt)
#cv2.imwrite("output.jpg",canny); 
#cv2.waitKey(0)
#cv2.destroyAllWindows()

"""
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)
 
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3
 
img = cv2.imread('D:/lion.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
cv2.namedWindow('canny demo')
 
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
 
CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
"""