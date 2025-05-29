# Simplified Sudoku Solver App
import os
import subprocess
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model with error handling
try:
    model = load_model("mnist_cnn.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Global counter for unique grid identification
grid_counter = 0


def preprocess_cell(cell_img):
    """Simple cell preprocessing for digit recognition"""
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    # Remove noise
    cell_img = cv2.medianBlur(cell_img, 3)

    # Apply threshold
    _, thresh = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 50:  # Too small, probably no digit
            return None

        # Get bounding box and extract digit
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_roi = thresh[y : y + h, x : x + w]

        # Resize to 28x28 for MNIST
        resized = cv2.resize(digit_roi, (28, 28))
        normalized = resized.astype("float32") / 255.0

        return normalized.reshape(1, 28, 28, 1)

    return None


def predict_digit(cell_img):
    """Predict digit from cell image"""
    if model is None:
        return 0

    try:
        processed = preprocess_cell(cell_img)
        if processed is None:
            return 0

        prediction = model.predict(processed, verbose=0)
        confidence = np.max(prediction)
        predicted_digit = int(np.argmax(prediction))

        # Only return digit if confidence is high and it's not 0
        if confidence > 0.7 and predicted_digit != 0:
            return predicted_digit
        else:
            return 0
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0


def extract_sudoku_grid(img):
    """Extract 9x9 Sudoku grid from image"""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Calculate cell dimensions
    height, width = thresh.shape
    cell_height = height // 9
    cell_width = width // 9

    # Extract digits from each cell
    grid = []
    for row in range(9):
        grid_row = []
        for col in range(9):
            # Extract cell
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width

            cell = thresh[y1:y2, x1:x2]
            digit = predict_digit(cell)
            grid_row.append(digit)

        grid.append(grid_row)

    return grid


def save_grid_to_file(grid, filename):
    """Save grid to text file"""
    with open(filename, "w") as f:
        for row in grid:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Grid saved to {filename}")


def call_cpp_solver():
    """Call the C++ Sudoku solver"""
    try:
        # Compile solver if needed (optional)
        if not os.path.exists("solver") and os.path.exists("solver.cpp"):
            print("Compiling solver.cpp...")
            compile_result = subprocess.run(
                ["g++", "-o", "solver", "solver.cpp"], capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                print(f"Compilation error: {compile_result.stderr}")
                return False

        # Run the solver
        print("Running C++ solver...")
        result = subprocess.run(
            ["./solver"], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print("Solver executed successfully!")
            return True
        else:
            print(f"Solver error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Solver timed out!")
        return False
    except Exception as e:
        print(f"Error calling solver: {e}")
        return False


def read_solution():
    """Read solution from solution.txt"""
    try:
        if os.path.exists("solution.txt"):
            with open("solution.txt", "r") as f:
                lines = f.readlines()

            solution = []
            for line in lines:
                row = [int(x) for x in line.strip().split()]
                if len(row) == 9:
                    solution.append(row)

            if len(solution) == 9:
                return solution
            else:
                print("Invalid solution format")
                return None
        else:
            print("solution.txt not found")
            return None
    except Exception as e:
        print(f"Error reading solution: {e}")
        return None


@app.route("/solve", methods=["POST"])
def solve_sudoku():
    """Main endpoint to solve Sudoku from image"""
    global grid_counter

    if model is None:
        return (
            jsonify({"error": "Model not loaded. Please train the model first."}),
            500,
        )

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read and process image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Extract Sudoku grid
        print("Extracting Sudoku grid from image...")
        grid = extract_sudoku_grid(img)

        # Create unique filename with timestamp and counter
        grid_counter += 1
        timestamp = int(time.time())
        input_filename = f"input_{timestamp}_{grid_counter}.txt"

        # Save grid to file
        save_grid_to_file(grid, input_filename)

        # Also save as input.txt for solver
        save_grid_to_file(grid, "input.txt")

        # Call C++ solver
        solver_success = call_cpp_solver()

        if not solver_success:
            return (
                jsonify(
                    {
                        "error": "Solver failed to execute",
                        "extracted_grid": grid,
                        "input_file": input_filename,
                    }
                ),
                500,
            )

        # Read solution
        solution = read_solution()

        if solution is None:
            return (
                jsonify(
                    {
                        "error": "Failed to read solution",
                        "extracted_grid": grid,
                        "input_file": input_filename,
                    }
                ),
                500,
            )

        return jsonify(
            {
                "status": "success",
                "extracted_grid": grid,
                "solution": solution,
                "input_file": input_filename,
                "message": "Sudoku solved successfully!",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "solver_exists": os.path.exists("solver") or os.path.exists("solver.cpp"),
        }
    )


if __name__ == "__main__":
    print("=== Simplified Sudoku Solver ===")
    if model is None:
        print("WARNING: Model not loaded! Please run the training script first.")

    if not os.path.exists("solver") and not os.path.exists("solver.cpp"):
        print(
            "WARNING: solver.cpp not found! Please ensure your C++ solver is available."
        )

    print("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
