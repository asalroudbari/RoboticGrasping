import cv2
from functions import *

img = cv2.imread("image_0.jpg")
scene = cv2.imread("image_1.jpg")
#Dout = place2(scene, img)
output = [["pglass", "pglass"], [[521,267], [217,170]], [0,0]]  
graspObjects(output) 


