import base64
import os
import subprocess
import sys
import uuid

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["file"]

    solver_dir = os.path.join(os.path.dirname(__file__), "development")

    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_input.png"
    output_filename = f"{unique_id}_output.png"

    input_path = os.path.join(solver_dir, input_filename)
    output_path = os.path.join(solver_dir, output_filename)

    try:
        file.save(input_path)

        command = [sys.executable, "main.py", input_path, output_path]

        subprocess.run(
            command,
            cwd=solver_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )

        with open(output_path, "rb") as f:
            img_bytes = f.read()

        img_str = base64.b64encode(img_bytes).decode("utf-8")
        return jsonify({"image": "data:image/png;base64," + img_str})

    except subprocess.CalledProcessError as e:
        print("Solver script failed:", e.stderr)
        return jsonify({"error": f"The solver script failed: {e.stderr}"}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    app.run(debug=True)
