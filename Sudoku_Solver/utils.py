import cv2
import numpy as np
from tensorflow.keras.models import load_model


def initializePredictModel():
    model = load_model("./model/sudoku_digit_model.h5")
    return model


def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, 1, 1, 11, 2
    )  # adaptive thresholding

    return imgThreshold


# finding the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    maxArea = 0
    for i in contours:
        area = cv2.contourArea(i)  # check the area of each contour
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(
                i, 0.02 * perimeter, True
            )  # approximate the corners
            if (
                area > maxArea and len(approx) == 4
            ):  # only finding the shapes which are rectangles or squares
                biggest = approx
                maxArea = area
    return biggest, maxArea


# splitting the boxes:
def splitBoxes(img):
    rows = np.vsplit(img, 9)  # splitting the image into 9 rows
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)  # splitting each row into 9 columns
        for box in cols:
            boxes.append(box)
    return boxes


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # top left
    myPointsNew[3] = myPoints[np.argmax(add)]  # bottom right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # top right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # bottom left
    return myPointsNew.astype(np.int32)


def getPrediction(boxes, model):
    result = []
    for image in boxes:
        # PREPARING THE IMAGE FOR PREDICTION
        img = np.asarray(image)
        img = img[4 : img.shape[0] - 4, 4 : img.shape[1] - 4]  # removing borders
        img = cv2.resize(img, (28, 28))  # resizing to 28x28
        img = img / 255.0  # normalizing the image
        img = img.reshape(1, 28, 28, 1)

        # GETTING THE PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        # print(f"Predicted class: {classIndex}, Probability: {probabilityValue}")
        # SAVING THE RESULT
        if probabilityValue > 0.8:  # threshold for confidence
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(
                    img,
                    str(numbers[(y * 9) + x]),
                    (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return img


def stackedImages(imgArray, scale=1):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = int(imgArray[0][0].shape[1] * scale)
    height = int(imgArray[0][0].shape[0] * scale)

    imgBlank = np.zeros((height, width, 3), np.uint8)

    if rowsAvailable:
        for i in range(rows):
            for j in range(cols):
                imgArray[i][j] = cv2.resize(imgArray[i][j], (0, 0), None, scale, scale)
                if len(imgArray[i][j].shape) == 2:  # if grayscale
                    imgArray[i][j] = cv2.cvtColor(imgArray[i][j], cv2.COLOR_GRAY2BGR)
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows
        hor_con = [imgBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
