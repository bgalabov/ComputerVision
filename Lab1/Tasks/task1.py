import cv2 as cv
import sys

if sys.argv[1:] is not None: 
    fileName = sys.argv[1]
else:
    fileName = "apple.jpg"

print(fileName)

img = cv.imread(cv.samples.findFile(fileName), cv.IMREAD_GRAYSCALE)
dst = None

if img is None:
    sys.exit("Could not read the image.")

dst = cv.GaussianBlur(img, (3, 3), 1.5)

grad_x = cv.Sobel(dst, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(dst, cv.CV_32F, 0, 1, ksize=3)

magnitude = cv.convertScaleAbs(cv.magnitude(grad_x, grad_y))

_, dst = cv.threshold(magnitude, 85, 255, cv.THRESH_BINARY)

cv.imwrite(fileName.split(".")[0] + "_edges.jpg", dst)

