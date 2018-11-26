import cv2
import numpy as np
from matplotlib import pyplot as plt 

template = cv2.imread('tmp.png')

for i in range(1, 26):
	img_name = 'test'+str(i)+'.jpg'
	img = cv2.imread(img_name)
	res = cv2.matchTemplate(img, template, eval('cv2.TM_CCORR_NORMED'))

	w,h = template[:,:,0].shape[::-1]

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	top_left = max_loc
	bottom_right = (top_left[0]+w, top_left[1]+h)

	imgplt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	cv2.rectangle(imgplt, top_left, bottom_right, 255, 2)

	out_name = 'rst'+str(i)+'.png'

	cv2.imwrite(out_name, imgplt)
#plt.imshow(imgplt)
#plt.title('Test'), plt.xticks([]), plt.yticks([])
#plt.show()
