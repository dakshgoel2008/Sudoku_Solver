import os
import subprocess
import sys

import cv2 as cv
import numpy as np
from utils import *

# TODO: replacing this with the backend logic
imagePath = os.path.join(os.getcwd(), "Sudoku_Solver", "images", "img_1.jpg")
imgHeight = 450
imgWidth = 450
# print(f"Image path: {imagePath}")

model = initializePredictModel()  # My Model

img = cv.imread(imagePath)

if img is None:
    print(f"Error: Could not load image from {imagePath}")
    sys.exit(1)

img = cv.resize(img, (imgWidth, imgHeight))
imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)
imgThreshold = preProcess(img)

"""Contours"""
imgCountours = img.copy()
imgBigCountour = img.copy()
contours, hierachy = cv.findContours(
    imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)  # RETR_EXTERNAL -> for returning only the extreme outer contours. CHAIN_APPROX_SIMPLE -> for squeezing contours.
cv.drawContours(imgCountours, contours, -1, (0, 255, 0), 3)  # draw the green contours.

# Find the biggest contour -> biggest contour is nothing but our grid of sudoku.
biggest = biggestContour(contours)
if biggest.size != 0:
    # biggest 4 point contour is our grid -> recognised by red color polka dots here.
    biggest = reorder(biggest)  # [top-left, top-right, bottom-left, bottom-right]
    cv.drawContours(imgBigCountour, biggest, -1, (0, 0, 255), 20)  # BGR

    # Warp perspective -> just stretching and squeezing the image for bird's eye view
    # just to correct the image angle errors via camera.
    # TODO: Have to improve it for more accuracy and good user experience
    pts1 = np.float32(biggest)
    pts2 = np.float32(
        [[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]]
    )  # Order in which we are considering the coordinates
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgWarpedColored = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
    imgDetectedDigits = imgBlank.copy()
    imgWarpedColored = cv.cvtColor(imgWarpedColored, cv.COLOR_BGR2GRAY)

    # Split and predict digits
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpedColored)  # split the warped Image into 81 boxes
    numbers = getPrediction(boxes, model, debug=True)
    print(f"Detected Numbers: {numbers}")  # will give the predicted numbers

    # displaying the detected numbers.
    imgDetectedDigits = displayNumbers(imgSolvedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 1, 0)  # means detected. else not detected.

    # Validate the detected board
    if len(numbers) != 81:
        print(f"Error: Expected 81 numbers, got {len(numbers)}")
        sys.exit(1)

    # Save board to input file
    flat_board = numbers.flatten()  # saving it as a 1D array
    # Convert all elements to regular Python integers
    flat_board = [int(x) for x in flat_board]  # converting to int each element

    input_file = os.path.join("Sudoku_Solver", "input_board.txt")  # path to input file
    output_file = os.path.join(
        "Sudoku_Solver", "output_board.txt"
    )  # path to output file

    try:
        with open(input_file, "w") as f:
            for i in range(9):
                # considering [0-8] rows and [0-8] cols.
                row = flat_board[i * 9 : (i + 1) * 9]
                f.write(" ".join(map(str, row)) + "\n")
        # print(f"Board saved to {input_file}")

        # Print the board being sent to solver for debugging
        print("\nBoard being sent to solver:")
        for i in range(9):
            row = flat_board[i * 9 : (i + 1) * 9]
            print(row)

    except Exception as e:
        print(f"Error saving input board: {e}")
        sys.exit(1)

    # Call the C++ executable
    solver_path = os.path.join(os.getcwd(), "Sudoku_Solver", "solver.exe")

    # Check if solver exists
    """ Integration steps of C++ solver with python and converting the got sudoku back in the image format """
    if not os.path.exists(solver_path):
        print(f"Error: Solver executable not found at {solver_path}")
        print("Make sure to compile your solver.cpp first:")
        print("g++ -o solver.exe solver.cpp")
        sys.exit(1)

    try:
        print("Calling C++ solver...")
        result = subprocess.run(
            [solver_path],
            check=True,  # Check for errors if solver exists or not.
            capture_output=True,  # capturing stdout and stderr
            text=True,  # just returning the stdout and stderr
            timeout=10,  # Reduced timeout to 10 seconds
            cwd=os.path.dirname(solver_path),
        )
        print("Solver completed successfully")  # completed bro
        # if result.stdout:
        #     print(f"Solver output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Solver failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Solver timed out (took more than 10 seconds)")
        print("This might indicate: some error")
        print("1. The solver is stuck in an infinite loop")
        print("2. The input board is invalid or very difficult")
        sys.exit(1)
    except Exception as e:
        print(f"Error running solver: {e}")
        sys.exit(1)

    # Read the solved board
    try:
        if not os.path.exists(output_file):
            print(f"Error: Output file {output_file} was not created by solver")
            sys.exit(1)

        solved_board = []
        with open(output_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = list(map(int, line.strip().split()))
                    if len(row) != 9:
                        print(
                            f"Error: Row {line_num} has {len(row)} numbers, expected 9"
                        )
                        sys.exit(1)
                    solved_board.append(row)
                except ValueError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    sys.exit(1)

        if len(solved_board) != 9:
            print(f"Error: Expected 9 rows, got {len(solved_board)}")
            sys.exit(1)

        print("\nOriginal Board:")
        original_board = numbers.reshape(9, 9)
        for row in original_board:
            print([int(x) for x in row])

        print("\nSolved Board:")
        for row in solved_board:
            print(row)

        # Convert to NumPy array for further processing
        solved_board = np.array(solved_board)

        # Create the overlay image showing only solved digits
        imgSolvedOverlay = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
        solved_numbers = solved_board.flatten()
        imgSolvedOverlay = displaySolutionNumbers(
            imgSolvedOverlay, numbers, solved_numbers, color=(0, 0, 255)
        )

        # just for mapping the solution overlay back to original perspective
        matrix_inv = cv.getPerspectiveTransform(pts2, pts1)

        # Create the solution overlay on warped image first
        imgSolutionWarped = imgBlank.copy()
        imgSolutionWarped = displaySolutionNumbers(
            imgSolutionWarped, numbers, solved_numbers, color=(0, 255, 0)
        )

        # Warp the solution back to original perspective
        imgSolutionOriginal = cv.warpPerspective(
            imgSolutionWarped, matrix_inv, (imgWidth, imgHeight)
        )

        # Combine original image with solution overlay
        imgFinalResult = img.copy()
        # Create mask for non-zero pixels in the solution overlay
        mask = cv.cvtColor(imgSolutionOriginal, cv.COLOR_BGR2GRAY)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        # Apply the overlay
        # colors = [0, 0, 255]
        for c in range(3):
            imgFinalResult[:, :, c] = np.where(
                # mask == 1, color[c], imgFinalResult[:, :, c]
                mask == 1,
                imgSolutionOriginal[:, :, c],
                imgFinalResult[:, :, c],
            )

        # Also create image showing solved digits on warped perspective
        imgSolvedDigits = displayNumbers(
            imgBlank.copy(), solved_numbers, color=(255, 255, 0)
        )

    except Exception as e:
        print(f"Error reading solved board: {e}")
        sys.exit(1)

else:
    print("No Sudoku board detected in the image")
    imgSolvedDigits = imgBlank.copy()
    imgSolvedOverlay = imgBlank.copy()
    imgFinalResult = img.copy()


""" Just for debugging purposes """
# Display images - Updated to show the overlay results
imgArray = (
    [img, imgThreshold, imgCountours, imgBigCountour],
    [imgDetectedDigits, imgSolvedDigits, imgSolvedOverlay, imgFinalResult],
)
stacked_images = stackedImages(imgArray, 1)
cv.namedWindow("Daksh Goel", cv.WINDOW_NORMAL)
cv.imshow("Daksh Goel", stacked_images)

# Optional: Save the final result
cv.imwrite("finalAnswer.jpg", imgFinalResult)
print("Final result saved as 'finalAnswer.jpg'")

cv.waitKey(0)
cv.destroyAllWindows()
