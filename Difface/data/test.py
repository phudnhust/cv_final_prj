import cv2 
img = cv2.imread('./targ/00182.png')
blur = cv2.GaussianBlur(img,(23,23),0)
cv2.imwrite('a.png',blur)
