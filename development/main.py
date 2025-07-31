import os
import subprocess
import sys

import cv2 as cv
import numpy as np
from utils import *

if len(sys.argv) != 3:
    print("Usage: python main.py <input_path> <output_path>")
    sys.exit(1)
imagePath = sys.argv[1]
outputPath = sys.argv[2]


imgHeight = 450
imgWidth = 450

model = initializePredictModel()
img = cv.imread(imagePath)

if img is None:
    print(f"Error: Could not load image from {imagePath}")
    sys.exit(1)

img = cv.resize(img, (imgWidth, imgHeight))
imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)
imgThreshold = preProcess(img)

imgCountours = img.copy()
imgBigCountour = img.copy()
contours, hierachy = cv.findContours(
    imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)
biggest = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv.drawContours(imgBigCountour, biggest, -1, (0, 0, 255), 20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgWarpedColored = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
    imgDetectedDigits = imgBlank.copy()
    imgWarpedColored = cv.cvtColor(imgWarpedColored, cv.COLOR_BGR2GRAY)
    boxes = splitBoxes(imgWarpedColored)
    numbers = getPrediction(boxes, model, debug=False)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    if len(numbers) != 81:
        sys.exit(1)
    flat_board = [int(x) for x in numbers.flatten()]
    input_file = "input_board.txt"
    output_file = "output_board.txt"
    with open(input_file, "w") as f:
        for i in range(9):
            row = flat_board[i * 9 : (i + 1) * 9]
            f.write(" ".join(map(str, row)) + "\n")
    solver_path = "solver.exe"
    if not os.path.exists(solver_path):
        sys.exit(1)
    try:
        subprocess.run(
            [solver_path], check=True, capture_output=True, text=True, timeout=10
        )
    except Exception as e:
        sys.exit(1)
    if not os.path.exists(output_file):
        sys.exit(1)
    solved_board = []
    with open(output_file, "r") as f:
        for line in f:
            solved_board.append(list(map(int, line.strip().split())))
    solved_board = np.array(solved_board)
    matrix_inv = cv.getPerspectiveTransform(pts2, pts1)
    imgSolutionWarped = imgBlank.copy()
    imgSolutionWarped = displaySolutionNumbers(
        imgSolutionWarped, numbers, solved_board.flatten(), color=(0, 255, 0)
    )
    imgSolutionOriginal = cv.warpPerspective(
        imgSolutionWarped, matrix_inv, (imgWidth, imgHeight)
    )
    imgFinalResult = img.copy()
    mask = cv.cvtColor(imgSolutionOriginal, cv.COLOR_BGR2GRAY)
    imgFinalResult[mask > 0] = imgSolutionOriginal[mask > 0]
else:
    imgFinalResult = img.copy()  # If no board found, return original image

cv.imwrite(outputPath, imgFinalResult)
print(f"Final result saved as '{outputPath}'")

if os.path.exists("input_board.txt"):
    os.remove("input_board.txt")
if os.path.exists("output_board.txt"):
    os.remove("output_board.txt")
