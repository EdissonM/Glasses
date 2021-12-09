import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import faceBlendCommon as fbc

# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

im = cv2.imread("girl-no-makeup.jpg")

RGB_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

points = fbc.getLandmarks(faceDetector, landmarkDetector, RGB_image)

glasses = cv2.imread("glasses.png")
lenses = cv2.imread("lenses.png")
width = int(np.abs(points[16][0] - points[0][0]))
height = int((glasses.shape[0] * width) / glasses.shape[1])
x = points[28][0] - int(width / 2)
y = points[20][1]
im = im.astype(float) / 255

lenses = cv2.resize(lenses, (width, height), cv2.INTER_CUBIC)
_, lenses_mask = cv2.threshold(cv2.bitwise_not(lenses)[:, :, 0], 10, 255, cv2.THRESH_BINARY)

lenses_mask = lenses_mask.astype(float) / 255
lenses_mask = cv2.merge((lenses_mask, lenses_mask, lenses_mask))
foreground_lenses = cv2.multiply(lenses_mask, lenses.astype(float) / 255)
cv2.imwrite("results/lenses_mask.jpg", (lenses_mask * 255).astype(np.uint8))

cv2.imwrite("results/foreground_lenses.jpg", (foreground_lenses * 255).astype(np.uint8))
im_lenses = im.copy()
background_lenses = cv2.multiply(1.0 - lenses_mask, im_lenses[y:y + height, x: x + width])

im_lenses[y:y + height, x: x + width] = cv2.add(foreground_lenses, background_lenses)

im = cv2.addWeighted(im_lenses, 0.5, im, 0.5, 0)
cv2.imwrite("results/lenses_image.jpg", (im * 255).astype(np.uint8))
glasses = cv2.resize(glasses, (width, height), cv2.INTER_CUBIC)
_, glasses_mask = cv2.threshold(cv2.bitwise_not(glasses)[:, :, 0], 10, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)

glasses_mask = cv2.erode(glasses_mask, kernel, iterations=1)
glasses_mask = cv2.GaussianBlur(glasses_mask, (3, 3), cv2.BORDER_DEFAULT)
glasses_mask = glasses_mask.astype(float) / 255
glasses_mask = cv2.merge((glasses_mask, glasses_mask, glasses_mask))
foreground = cv2.multiply(glasses_mask, glasses.astype(float) / 255)
cv2.imwrite("results/glasses_mask.jpg", (glasses_mask * 255).astype(np.uint8))

cv2.imwrite("results/foreground_glasses.jpg", (foreground * 255).astype(np.uint8))
# Multiply the background with ( 1 - alpha )

background = cv2.multiply(1.0 - glasses_mask, im[y:y + height, x: x + width])

im[y:y + height, x: x + width] = cv2.add(foreground, background)
plt.imshow(cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("results/Glasses_image.jpg", (im * 255).astype(np.uint8))
