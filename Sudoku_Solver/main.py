# Let' s begin bro
import os

from utils import *

imagePath = os.path.join(os.getcwd(), "Sudoku_Solver", "images", "sudoku1.jpg")
imgHeight = 300
imgWidth = 300
# print(f"Image path: {imagePath}")
model = initializePredictModel()  # loading my CNN based model -> TODO

# Lets prepare the image:
img = cv2.imread(imagePath)
img = cv2.resize(img, (imgWidth, imgHeight))
imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)  # just for debugging
imgThreshold = preProcess(img)

# Finding the countours:
imgCountours = img.copy()  # copying for display
imgBigCountour = img.copy()
contours, hierachy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(imgCountours, contours, -1, (0, 255, 0), 3)


# finding the biggest countour:
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(
        imgBigCountour, biggest, -1, (0, 0, 255), 20
    )  # drawing the biggest contour
    pts1 = np.float32(biggest)  # preparing the points for WARP -> image wrapping
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(
        pts1, pts2
    )  # getting the transformation matrix
    imgWarpedColored = cv2.warpPerspective(
        img, matrix, (imgWidth, imgHeight)
    )  # warping the image
    imgDetectedDigits = imgBlank.copy()
    imgWarpedColored = cv2.cvtColor(imgWarpedColored, cv2.COLOR_BGR2GRAY)

    # splitting the image and finding each digit:
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpedColored)
    numbers = getPrediction(boxes, model)
    print(f"Numbers: {numbers}")
    imgDetectedDigits = displayNumbers(imgSolvedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 1, 0)  # creating a mask for the digits

# to show the images in a single window:
imgArray = (
    [img, imgThreshold, imgCountours, imgBigCountour],
    [imgDetectedDigits, imgBlank, imgBlank, imgBlank],
)
stacked_images = stackedImages(imgArray, 1)
cv2.imshow("Stacked Images", stacked_images)

cv2.waitKey(0)  # Wait for a key press to close the window
