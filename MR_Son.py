import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import*

# Reading in and displaying our image
from PIL import Image

images = map(Image.open, ['000000.jpg', '000001.jpg', '000002.jpg','000003.jpg','000004.jpg'])
#list_im = ['000000.jpg','000002.jpg','000003.jpg','000004.jpg']
new_im = Image.new('RGB', (1250,250)) #creates a new empty image, RGB mode, and size 444 by 95

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.jpg')

"""
for elem in list_im:
    for i in range(0,4000,250):
        im=Image.open(elem)
        new_im.paste(im, (i,0))
"""



"""

import sys
from PIL import Image

images = map(Image.open, ['000000.jpg', '000001.jpg', '000002.jpg'])
widths, heights = Zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test1.jpg')

"""


"""
[X,map]=cv2.imread('1.Hasta -AX T2 FSE-23822')              
montage(X,map,'Size',[6,6])
cv2.imshow(X,[])
cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows # close the img window.
"""


"""
image = cv2.imread('000002.jpg')
cv2.imshow('Original', image)
# Create our shapening kernel, it must equal to one eventually


img_median = cv2.medianBlur(image, 5) # Add median filter to image         #GÖRÜNTÜ BULANIKLAŞTIRMA

cv2.imshow('median filter', img_median) # Display img with median filter
cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows # close the img window.



ret,thresh1 = cv2.threshold(img_median,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img_median,127,255,cv2.THRESH_BINARY_INV)                   #THRESHOLDİNG İŞLEMİİ
ret,thresh3 = cv2.threshold(img_median,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img_median,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img_median,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):                                                                          
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show() 

kernelMat = np.ones((15,15),np.uint8)

opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernelMat)
opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernelMat)
opening3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernelMat)
opening4= cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernelMat)                              #MORFOLOJİK İŞLEMLER
opening5= cv2.morphologyEx(thresh5, cv2.MORPH_OPEN, kernelMat)
titles2 = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images2 = [image, opening1, opening2, opening3, opening4, opening5]
for j in range(6):                                                                          
    plt.subplot(2,3,j+1),plt.imshow(images2[j],'gray')
    plt.title(titles[j])
    plt.xticks([]),plt.yticks([])
plt.show()


cv2.imshow("Opening", opening)
cv2.waitKey(0)
"""
