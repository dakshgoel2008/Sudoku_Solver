#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

class SudokuSolver {
private:
  vector<vector<int>> grid;
  int n;
  int sqrt_n;

public:
  SudokuSolver(int size) : n(size), sqrt_n(sqrt(size)) {
    grid.resize(n, vector<int>(n, 0));
  }

  bool isValid(int row, int col, int num) {
    // Check row
    for (int j = 0; j < n; j++) {
      if (grid[row][j] == num) {
        return false;
      }
    }

    // Check column
    for (int i = 0; i < n; i++) {
      if (grid[i][col] == num) {
        return false;
      }
    }

    // Check 3x3 box
    int box_row = (row / sqrt_n) * sqrt_n;
    int box_col = (col / sqrt_n) * sqrt_n;

    for (int i = box_row; i < box_row + sqrt_n; i++) {
      for (int j = box_col; j < box_col + sqrt_n; j++) {
        if (grid[i][j] == num) {
          return false;
        }
      }
    }

    return true;
  }

  bool solve() {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == 0) {
          for (int num = 1; num <= n; num++) {
            if (isValid(i, j, num)) {
              grid[i][j] = num;

              if (solve()) {
                return true;
              }

              grid[i][j] = 0; // backtrack
            }
          }
          return false;
        }
      }
    }
    return true; // All cells filled
  }

  bool isValidSudoku() {
    // Check if the initial configuration is valid
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] != 0) {
          int temp = grid[i][j];
          grid[i][j] = 0; // Temporarily remove

          if (!isValid(i, j, temp)) {
            grid[i][j] = temp; // Restore
            return false;
          }

          grid[i][j] = temp; // Restore
        }
      }
    }
    return true;
  }

  bool loadFromFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Cannot open input file " << filename << endl;
      return false;
    }

    file >> n;
    if (n != 9) {
      cerr << "Error: Only 9x9 sudoku puzzles are supported" << endl;
      return false;
    }

    sqrt_n = sqrt(n);
    grid.resize(n, vector<int>(n));

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        file >> grid[i][j];
        if (grid[i][j] < 0 || grid[i][j] > n) {
          cerr << "Error: Invalid digit " << grid[i][j] << " at position (" << i
               << "," << j << ")" << endl;
          return false;
        }
      }
    }

    file.close();
    return true;
  }

  bool saveToFile(const string &filename) {
    ofstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: Cannot create output file " << filename << endl;
      return false;
    }

    file << n << endl;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        file << grid[i][j];
        if (j < n - 1)
          file << " ";
      }
      file << endl;
    }

    file.close();
    return true;
  }

  void printGrid() {
    cout << "Sudoku Grid:" << endl;
    for (int i = 0; i < n; i++) {
      if (i % 3 == 0 && i != 0) {
        cout << "------+-------+------" << endl;
      }
      for (int j = 0; j < n; j++) {
        if (j % 3 == 0 && j != 0) {
          cout << "| ";
        }
        cout << grid[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  int countFilledCells() {
    int count = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] != 0) {
          count++;
        }
      }
    }
    return count;
  }
};

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
    cout << "Example: " << argv[0] << " puzzle.txt solution.txt" << endl;
    return 1;
  }

  string inputFile = argv[1];
  string outputFile = argv[2];

  SudokuSolver solver(9);

  // Load the puzzle
  if (!solver.loadFromFile(inputFile)) {
    return 1;
  }

  cout << "Loaded puzzle from " << inputFile << endl;
  cout << "Filled cells: " << solver.countFilledCells() << "/81" << endl;

  // Validate the initial configuration
  if (!solver.isValidSudoku()) {
    cout << "Error: Invalid sudoku configuration" << endl;
    return 1;
  }

  // Solve the puzzle
  auto start = chrono::high_resolution_clock::now();
  bool solved = solver.solve();
  auto end = chrono::high_resolution_clock::now();

  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

  if (solved) {
    cout << "Solution found in " << duration.count() << " milliseconds" << endl;

    if (solver.saveToFile(outputFile)) {
      cout << "Solution saved to " << outputFile << endl;
    }
  } else {
    cout << "No solution exists for this puzzle" << endl;
    return 1;
  }

  return 0;
}