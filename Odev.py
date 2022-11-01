# Emirhan Erdal 02205076041
import cv2
import numpy as np
import matplotlib.pyplot as plt
picture = cv2.imread("foto.jpg")
s = picture.shape
cv2.imshow('picture', picture)
picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
cv2.imshow('picture gray', picture_gray)
H = np.zeros(shape=(256,1))
for i in range(s[0]):
    for j in range(s[1]):
        k = picture_gray[i,j]
        H[k,0] = H[k,0]+1
plt.plot(H)
plt.show()
cv2.waitKey(0)
picture2 = cv2.imread('foto2.jpg')
cv2.imshow('picture', picture)
picture = cv2.imread('foto2.jpg',0)
cv2.imshow('picture-0', picture)
[h, w] = picture.shape
picture2 = np.zeros([h, w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        picture2[i, j] = 255 - picture[i, j]
cv2.imshow("Ters resim", picture)
cv2.waitKey()
