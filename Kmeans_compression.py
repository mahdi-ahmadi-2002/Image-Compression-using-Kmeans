import cv2 as cv
from sklearn.cluster import KMeans


def compressor(image, n_clusters):
    rows, columns, _ = image.shape
    image = image.reshape(rows * columns, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image)
    return kmeans.labels_.reshape(rows, columns), kmeans.cluster_centers_


def display(labels, n_clusters, title):
    compressed_image = n_clusters[labels]
    compressed_image = compressed_image.astype('uint8')
    cv.imshow(title, compressed_image)


img = cv.imread('image2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
centers = 2
labels_arr, centers_arr = compressor(img, centers)
display(labels_arr, centers_arr, 'BGR')
labels_arr, centers_arr = compressor(hsv, centers)
display(labels_arr, centers_arr, 'HSV')
labels_arr, centers_arr = compressor(lab, centers)
display(labels_arr, centers_arr, 'LAB')

cv.waitKey(0)
