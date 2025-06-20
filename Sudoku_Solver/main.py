import os
import subprocess
import sys

import cv2
import numpy as np
from utils import *

# I will replace it with the backend logic later
imagePath = os.path.join(
    os.getcwd(), "Sudoku_Solver", "images", "sudoku2.jpg"
)  # just a sample image for now
imgHeight = 450
imgWidth = 450
# print(f"Image path: {imagePath}")

# Initializing the CNN model
model = initializePredictModel()

# Preparing the image
img = cv2.imread(imagePath)

if img is None:  # fixing the issue of runtime error here yaar.
    print(f"Error: Could not load image from {imagePath}")
    sys.exit(1)

img = cv2.resize(img, (imgWidth, imgHeight))  # resizing the window
imgBlank = np.zeros(
    (imgHeight, imgWidth, 3), np.uint8
)  # blank image which will be overlayed later as the image are processed.
imgThreshold = preProcess(img)  # see utils.py

# Find contours
imgCountours = img.copy()
imgBigCountour = img.copy()
# just finding the contours in the binary image.
contours, hierachy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)  # RETR_EXTERNAL -> returns only the extreme outer contours. CHAIN_APPROX_SIMPLE -> returns only the end points of the contours
cv2.drawContours(imgCountours, contours, -1, (0, 255, 0), 3)

# Find the biggest contour -> biggest contour is nothing but our grid of sudoku.
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    # biggest 4 point contour is our grid -> recognised by red color polka dots here.
    biggest = reorder(biggest)  # [top-left, top-right, bottom-left, bottom-right]
    cv2.drawContours(imgBigCountour, biggest, -1, (0, 0, 255), 20)

    # Warp perspective -> just stretching and squeezing the image dude for bird's eye view
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpedColored = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    imgDetectedDigits = imgBlank.copy()
    imgWarpedColored = cv2.cvtColor(imgWarpedColored, cv2.COLOR_BGR2GRAY)

    # Split and predict digits
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpedColored)
    numbers = getPrediction(boxes, model)
    print(f"Detected Numbers: {numbers}")

    imgDetectedDigits = displayNumbers(imgSolvedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 1, 0)

    # Validate the detected board
    if len(numbers) != 81:
        print(f"Error: Expected 81 numbers, got {len(numbers)}")
        sys.exit(1)

    # Save board to input file
    flat_board = numbers.flatten()
    # Convert all elements to regular Python integers
    flat_board = [int(x) for x in flat_board]

    input_file = os.path.join("Sudoku_Solver", "input_board.txt")
    output_file = os.path.join("Sudoku_Solver", "output_board.txt")

    try:
        with open(input_file, "w") as f:
            for i in range(9):
                row = flat_board[i * 9 : (i + 1) * 9]
                f.write(" ".join(map(str, row)) + "\n")
        print(f"Board saved to {input_file}")

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
    if not os.path.exists(solver_path):
        print(f"Error: Solver executable not found at {solver_path}")
        print("Make sure to compile your solver.cpp first:")
        print("g++ -o solver.exe solver.cpp")
        sys.exit(1)

    try:
        print("Calling C++ solver...")
        result = subprocess.run(
            [solver_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,  # Reduced timeout to 10 seconds
            cwd=os.path.dirname(solver_path),
        )  # Run in solver directory
        print("Solver completed successfully")
        if result.stdout:
            print(f"Solver output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Solver failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Solver timed out (took more than 10 seconds)")
        print("This might indicate:")
        print("1. The solver is stuck in an infinite loop")
        print("2. The input board is invalid or very difficult")
        print("3. There's an issue with the solver algorithm")
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

        # NOTE: **NEW CODE: Create overlay with only missing digits**
        def displaySolutionNumbers(
            img, originalNumbers, solvedNumbers, color=(0, 255, 0)
        ):
            """
            Display only the solved digits that were originally missing (0s)
            """
            secW = int(img.shape[1] / 9)
            secH = int(img.shape[0] / 9)

            for x in range(0, 9):
                for y in range(0, 9):
                    # Only display if the original position was empty (0)
                    if originalNumbers[y * 9 + x] == 0:
                        cv2.putText(
                            img,
                            str(int(solvedNumbers[y * 9 + x])),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            return img

        # Create the overlay image showing only solved digits
        imgSolvedOverlay = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
        solved_numbers = solved_board.flatten()
        imgSolvedOverlay = displaySolutionNumbers(
            imgSolvedOverlay, numbers, solved_numbers, color=(0, 255, 0)
        )

        # **Optional: Create inverse perspective to show on original image**
        # Get inverse transformation matrix
        matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)

        # Create the solution overlay on warped image first
        imgSolutionWarped = imgBlank.copy()
        imgSolutionWarped = displaySolutionNumbers(
            imgSolutionWarped, numbers, solved_numbers, color=(0, 255, 0)
        )

        # Warp the solution back to original perspective
        imgSolutionOriginal = cv2.warpPerspective(
            imgSolutionWarped, matrix_inv, (imgWidth, imgHeight)
        )

        # Combine original image with solution overlay
        imgFinalResult = img.copy()
        # Create mask for non-zero pixels in the solution overlay
        mask = cv2.cvtColor(imgSolutionOriginal, cv2.COLOR_BGR2GRAY)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        # Apply the overlay
        for c in range(3):
            imgFinalResult[:, :, c] = np.where(
                mask == 1, imgSolutionOriginal[:, :, c], imgFinalResult[:, :, c]
            )

        # Also create image showing solved digits on warped perspective
        imgSolvedDigits = displayNumbers(
            imgBlank.copy(), solved_numbers, color=(0, 255, 0)
        )

    except Exception as e:
        print(f"Error reading solved board: {e}")
        sys.exit(1)

else:
    print("No Sudoku board detected in the image")
    imgSolvedDigits = imgBlank.copy()
    imgSolvedOverlay = imgBlank.copy()
    imgFinalResult = img.copy()

# Display images - Updated to show the overlay results
imgArray = (
    [img, imgThreshold, imgCountours, imgBigCountour],
    [imgDetectedDigits, imgSolvedOverlay, imgBlank, imgBlank],
)
stacked_images = stackedImages(imgArray, 1)
cv2.namedWindow("Stacked Images", cv2.WINDOW_NORMAL)
cv2.imshow("Stacked Images", stacked_images)

# Optional: Save the final result
cv2.imwrite("sudoku_solved_overlay.jpg", imgSolvedOverlay)
print("Final result saved as 'sudoku_solved_overlay.jpg'")

cv2.waitKey(0)
cv2.destroyAllWindows()
