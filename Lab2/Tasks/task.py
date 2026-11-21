import sys
import math
import cv2 as cv
import numpy as np
import random as rng
rng.seed(12345)

def main(argv):

    
    default_file = 'stuff.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    canny_output = cv.Canny(src, 150, 200, None, 3)

    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)


    for i, c in enumerate(contours):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            # contour
            cv.drawContours(drawing, contours, i, color)

     # Get the moments
    mu = [None]*len(contours)
    mc = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])

    for i, c in enumerate(contours):

        #centroid (center of mass))
        cx = int(mu[i]['m10']/(mu[i]['m00'] + 1e-5))
        cy = int(mu[i]['m01']/(mu[i]['m00'] + 1e-5))

        #bounding rectangle (i.e. width and height)
        x, y, w, h = cv.boundingRect(contours[i])

        #aspect ratio
        aspect_ratio = float(w)/h

        #contour area
        area = cv.contourArea(contours[i])
        rect_area = w*h
        extent = float(area)/rect_area

        if area > 0:
            hull = cv.convexHull(contours[i])
            hull_area = cv.contourArea(hull)
            solidity = float(area)/hull_area

        #equivalent diameter
        eq_diameter = np.sqrt(4*area/np.pi)

        #orientation
        if c.shape[0] >= 5:
            (x,y),(MA,ma),angle = cv.fitEllipse(contours[i])
        else:
             angle = 0


        # draw mass centers and put numbers for each object
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.circle(drawing, (cx - 4 , cy), 2, color, -1)
        cv.putText(drawing, str(i), (cx, cy),
                cv.FONT_HERSHEY_PLAIN, 1, color, 1)

        
        print(' * Contour[%d] - Center(x, y) = (%d, %d) - Aspect Ratio: %.2f '
              '- Extent: %.2f - Solidity: %.2f - Eq-Diameter: %.2f - Orientation: %.2f' % 
              (i, cx, cy, aspect_ratio, extent, solidity, eq_diameter, angle))

    
    cv.imshow('Contours', drawing)

    cv.waitKey()
    return 0



        



if __name__ == "__main__":
    main(sys.argv[1:])