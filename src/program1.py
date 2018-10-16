import cv2

template = cv2.imread('./images/template.jpg')
source = cv2.imread('./images/source.jpg')

(tempH, tempW) = template.shape[:2]

# find the template in the source image
result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)

# draw the bounding box on the source image
cv2.rectangle(source, (x, y), (x + tempW, y + tempH), (0, 255, 0), 2)

# show the images
cv2.imshow("Source", source)
cv2.imshow("Template", template)
cv2.waitKey(0)