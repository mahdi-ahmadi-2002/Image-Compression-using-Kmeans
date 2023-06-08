import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def compressor(image, n_clusters):
    rows, columns, _ = image.shape
    image = image.reshape(rows * columns, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image)
    labels = kmeans.labels_.reshape(rows, columns)
    n_clusters = kmeans.cluster_centers_
    compressed_image = n_clusters[labels]
    compressed_image = compressed_image.astype('uint8')
    return compressed_image


img = cv.imread('Resources/Photos/img2.jpg')
width = int(img.shape[1] * 0.5)
height = int(img.shape[0] * 0.5)

dimensions = (width, height)

img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

cv.imshow('img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 50, 25)
ret, thresh1 = cv.threshold(canny, 100, 255, cv.THRESH_BINARY_INV)
thresh1 = cv.cvtColor(thresh1, cv.COLOR_GRAY2BGR)

cv.imshow('thresh1', thresh1)
blur_img = cv.blur(img, (7, 7))

k_clustered = compressor(img, 8)

color = cv.bilateralFilter(img, 31, 250, 250)

cartoon_blur = cv.bitwise_and(blur_img, thresh1)
cartoon_cluster = cv.bitwise_and(k_clustered, thresh1)
cartoon_bilateral = cv.bitwise_and(color, thresh1)

cv.imshow('cartoon_blur', cartoon_blur)
cv.imshow('cartoon_cluster', cartoon_cluster)
cv.imshow('cartoon_bilateral', cartoon_bilateral)

cv.waitKey(0)
