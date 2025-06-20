import os
import subprocess
import sys

from utils import *

imagePath = os.path.join(os.getcwd(), "Sudoku_Solver", "images", "sudoku2.jpg")
imgHeight = 450
imgWidth = 450
print(f"Image path: {imagePath}")

# Initialize the CNN model
model = initializePredictModel()

# Prepare the image
img = cv2.imread(imagePath)
if img is None:
    print(f"Error: Could not load image from {imagePath}")
    sys.exit(1)

img = cv2.resize(img, (imgWidth, imgHeight))
imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)
imgThreshold = preProcess(img)

# Find contours
imgCountours = img.copy()
imgBigCountour = img.copy()
contours, hierachy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(imgCountours, contours, -1, (0, 255, 0), 3)

# Find the biggest contour
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigCountour, biggest, -1, (0, 0, 255), 20)

    # Warp perspective
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

        # Create image showing solved digits
        solved_numbers = solved_board.flatten()
        imgSolvedDigits = displayNumbers(
            imgBlank.copy(), solved_numbers, color=(0, 255, 0)
        )

        # Clean up temporary files (optional)
        # os.remove(input_file)
        # os.remove(output_file)

    except Exception as e:
        print(f"Error reading solved board: {e}")
        sys.exit(1)

else:
    print("No Sudoku board detected in the image")
    imgSolvedDigits = imgBlank.copy()

# Display images
imgArray = (
    [img, imgThreshold, imgCountours, imgBigCountour],
    [imgDetectedDigits, imgSolvedDigits, imgBlank, imgBlank],
)
stacked_images = stackedImages(imgArray, 1)
cv2.namedWindow("Stacked Images", cv2.WINDOW_NORMAL)
cv2.imshow("Stacked Images", stacked_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
