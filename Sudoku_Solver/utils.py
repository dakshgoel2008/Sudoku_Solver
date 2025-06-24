import os

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model


def initializePredictModel():
    """Initializing the model"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # best_model.h5 is the best model and final_model.h5 is the final model after training
    # just created for callback and tried to use early stopping for better results.
    model_path = os.path.join(base_dir, "..", "CNN_Digit_Classifier", "best_model.h5")

    model_path = os.path.abspath(model_path)

    print("Loading model from:", model_path)
    model = load_model(model_path)
    return model


def preProcess(img):
    """function for preprocessing the image"""
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    # converting to binary image!
    imgThreshold = cv.adaptiveThreshold(
        imgBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2
    )
    return imgThreshold


# finding the biggest contour
def biggestContour(contours):
    """Expecting that biggest contour will be the grid itself"""
    biggest = np.array([])
    maxArea = 0
    for i in contours:
        area = cv.contourArea(i)  # check the area of each contour
        # TODO: will make it adaptive to image resolution later.
        if area > 50:
            perimeter = cv.arcLength(i, True)
            approx = cv.approxPolyDP(
                i, 0.02 * perimeter, True
            )  # approximate the corners -> more likely the image will be a recangle or square
            if (
                area > maxArea and len(approx) == 4
            ):  # only finding the shapes which are rectangles or squares
                biggest = approx
                maxArea = area
    return biggest  # no need to send maxArea


# splitting the boxes:
def splitBoxes(img):
    """Splitting in 81 boxes for obvious reasons"""
    rows = np.vsplit(img, 9)  # splitting the image into 9 rows
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)  # splitting each row into 9 columns
        for box in cols:
            boxes.append(box)
    return boxes


def reorder(myPoints):
    """Function to avoid random order of corner points in the contour"""
    """Here I am ordering them as [top-left, top-right, bottom-left, bottom-right]"""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # top left # min value
    # adding will give the diagonal element
    myPointsNew[3] = myPoints[np.argmax(add)]  # bottom right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # top right  # min value
    myPointsNew[2] = myPoints[np.argmax(diff)]  # bottom left # diagonal element
    return myPointsNew.astype(np.int32)


# TODO: Start from here
def isEmptyCell(box, debug=False):
    """Empty cell detection"""
    # Remove border pixels more aggressively
    h, w = box.shape
    inner_box = box[6 : h - 6, 6 : w - 6] if h > 12 and w > 12 else box

    # Method 1: Check white pixel density
    total_pixels = inner_box.size
    white_pixels = np.sum(inner_box > 200)  # Count bright pixels
    white_ratio = white_pixels / total_pixels

    # Method 2: Check for significant contours
    # Apply threshold to get binary image
    _, binary = cv.threshold(inner_box, 127, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Calculate significant contour area
    significant_contour_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 10:  # Only count contours with reasonable size
            significant_contour_area += area

    contour_ratio = significant_contour_area / total_pixels

    # Method 3: Check standard deviation (empty cells have low variation)
    std_dev = np.std(inner_box.astype(np.float32))

    # Method 4: Check connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary)

    # Count significant components (excluding background)
    significant_components = 0
    for i in range(1, num_labels):
        component_area = stats[i, cv.CC_STAT_AREA]
        if component_area > total_pixels * 0.02:  # At least 2% of cell area
            significant_components += 1

    if debug:
        print(
            f"White ratio: {white_ratio:.3f}, Contour ratio: {contour_ratio:.3f}, "
            f"Std dev: {std_dev:.1f}, Significant components: {significant_components}"
        )

    # Cell is empty if:
    # 1. Low white pixel ratio (mostly dark/background)
    # 2. Very low contour area
    # 3. Low standard deviation (uniform appearance)
    # 4. No significant connected components
    is_empty = (
        white_ratio < 0.15
        and contour_ratio < 0.08
        and std_dev < 25
        and significant_components <= 1
    )

    return is_empty


def preprocessCellForModel(box):
    """Preprocess individual cell for better model prediction"""
    # Remove more border pixels to eliminate grid lines
    h, w = box.shape
    img = box[6 : h - 6, 6 : w - 6] if h > 12 and w > 12 else box

    # Apply additional morphological operations to clean up the digit
    kernel = np.ones((2, 2), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # Resize to model input size
    img = cv.resize(img, (32, 32))

    # Apply additional Gaussian blur to smooth the digit
    img = cv.GaussianBlur(img, (3, 3), 0)

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = img.reshape(1, 32, 32, 1)

    return img


def getPrediction(boxes, model, debug=False):
    """prediction function"""
    result = []

    for i, image in enumerate(boxes):
        img = np.asarray(image)

        # First check if cell is empty using our improved detection
        if isEmptyCell(img, debug=debug and i < 5):  # Debug first 5 cells
            result.append(0)
            if debug and i < 10:
                print(f"Cell {i}: Detected as EMPTY")
            continue

        # Preprocess for model prediction
        processed_img = preprocessCellForModel(img)

        # Get model prediction
        predictions = model.predict(processed_img, verbose=0)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)

        if debug and i < 10:
            print(
                f"Cell {i}: Predicted digit {classIndex[0]}, confidence: {probabilityValue:.3f}"
            )

        # Use stricter confidence threshold and additional checks
        if probabilityValue > 0.85:  # Increased threshold
            predicted_digit = classIndex[0]

            # Additional validation: check if prediction makes sense
            # If confidence is marginal and predicted digit is 1 or 7 (common misclassifications),
            # do additional verification
            if probabilityValue < 0.95 and predicted_digit in [1, 7]:
                # Check if the cell actually has enough content to be a digit
                inner_box = img[6 : img.shape[0] - 6, 6 : img.shape[1] - 6]
                white_pixels = np.sum(inner_box > 200)
                total_pixels = inner_box.size

                if (
                    white_pixels / total_pixels < 0.2
                ):  # Too few white pixels for a digit
                    result.append(0)
                    if debug and i < 10:
                        print(
                            f"Cell {i}: Low confidence {predicted_digit} rejected as empty"
                        )
                else:
                    result.append(predicted_digit)
            else:
                result.append(predicted_digit)
        else:
            result.append(0)

    return result


def displayNumbers(img, numbers, color=(0, 0, 255)):
    """overlaying the digits to its correct place
    Just to enhance the beauty of the image😎
    """
    secW = int(img.shape[1] / 9)  # width of each cell
    secH = int(img.shape[0] / 9)  # height of each cell
    for x in range(0, 9):
        for y in range(0, 9):
            # numbers is a list of 81 no's
            if numbers[(y * 9) + x] != 0:  # means it is predicted (earlier it was 0)
                cv.putText(
                    img,
                    str(numbers[(y * 9) + x]),
                    (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                    # x*secW gives the x position to be putted, then move to the center by secW/2 and basic offset of 10 pixels for center alignment
                    # same goes for secH also.
                    cv.FONT_HERSHEY_COMPLEX_SMALL,  # Font
                    2,  # Font scale
                    color,  # color
                    2,  # Thickness
                    cv.LINE_AA,  # Line type(for smooth edges)
                )
    return img


def displaySolutionNumbers(img, originalNumbers, solvedNumbers, color=(0, 0, 255)):
    """
    Display only the solved digits that were originally missing (0s)
    """
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)

    for x in range(0, 9):
        for y in range(0, 9):
            # Only display if the original position was empty (0)
            if originalNumbers[y * 9 + x] == 0:
                cv.putText(
                    img,
                    str(int(solvedNumbers[y * 9 + x])),
                    (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    color,
                    2,
                    cv.LINE_AA,
                )
    return img


def stackedImages(imgArray, scale=1):
    """Just for debugging and seeing the current state I am working at"""
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = int(imgArray[0][0].shape[1] * scale)
    height = int(imgArray[0][0].shape[0] * scale)

    imgBlank = np.zeros((height, width, 3), np.uint8)

    if rowsAvailable:
        for i in range(rows):
            for j in range(cols):
                imgArray[i][j] = cv.resize(imgArray[i][j], (0, 0), None, scale, scale)
                if len(imgArray[i][j].shape) == 2:  # if grayscale
                    imgArray[i][j] = cv.cvtColor(imgArray[i][j], cv.COLOR_GRAY2BGR)
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows
        hor_con = [imgBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        # ver_con = np.concatenate(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
