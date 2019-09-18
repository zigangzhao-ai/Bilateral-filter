import cv2
import numpy
import matplotlib.pyplot as plt

#gussian filter
def gaussian(x,sigma):
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))


def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))

##bilateral_filter
def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = numpy.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])

                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
        print(row)
    return new_image

image = cv2.imread("lean.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


filtered_image_OpenCV = cv2.bilateralFilter(gray, 20, 200.0, 200.0)
cv2.imwrite("filtered_image_OpenCV50.png", filtered_image_OpenCV)

image_own = bilateral_filter(gray, 20, 200.0,200.0)
cv2.imwrite("filtered_image_own50.png", image_own)

"""
plt.figure()
plt.title('origin')
plt.imshow(image)
plt.show()


plt.figure()
plt.title('gray')
plt.imshow(gray)
plt.show()



plt.figure()
plt.title('Filter')
plt.imshow(filtered_image_OpenCV)
plt.show()


plt.figure()
plt.title('my_own')
plt.imshow(image_own)
plt.show()

"""

cv2.imshow('filter', filtered_image_OpenCV)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow('filtered', image_own)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
