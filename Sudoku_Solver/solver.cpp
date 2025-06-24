#include <bits/stdc++.h>
using namespace std;
#define fastio()                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL)

bool safeHai(vector<vector<int>> &a, int i, int j, int no, int n) {
  // 1. same no (number) row ya col mei nhi hona chahiye
  for (int k = 0; k < n; ++k) {
    if (a[k][j] == no || a[i][k] == no) {
      return false;
    }
  }

  // 2. "no" current 3x3 matrix mei nhi hona chahiye
  int rn = sqrt(n); // rn = root of n (3 for 9x9)
  int si = (i / rn) * rn;
  int sj = (j / rn) * rn;

  for (int r = si; r < si + rn; ++r) {
    for (int c = sj; c < sj + rn; ++c) {
      if (a[r][c] == no) {
        return false;
      }
    }
  }

  // Safe to place
  return true;
}

bool solveKaro(vector<vector<int>> &a, int i, int j, int n) {
  // base case
  if (i == n) {
    return true; // Solution found
  }

  // If column ends, go to next row
  if (j == n) {
    return solveKaro(a, i + 1, 0, n);
  }

  // If cell already filled, skip
  if (a[i][j] != 0) {
    return solveKaro(a, i, j + 1, n);
  }

  // Try filling 1 to n
  for (int no = 1; no <= n; ++no) {
    if (safeHai(a, i, j, no, n)) {
      a[i][j] = no;
      if (solveKaro(a, i, j + 1, n)) {
        return true;
      }
      a[i][j] = 0; // Backtrack
    }
  }

  return false;
}

int main() {
  fastio();

  // Reading data from file
  ifstream inFile("input_board.txt");
  if (!inFile) {
    cerr << "Error: Cannot open input_board.txt" << endl;
    return 1;
  }

  int n = 9;
  vector<vector<int>> board(n, vector<int>(n));

  // Read from file
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      inFile >> board[i][j];
    }
  }
  inFile.close();

  // Lets Print input board
  cout << "Input board:" << endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << board[i][j] << " ";
    }
    cout << endl;
  }

  // Solve the sudoku
  if (solveKaro(board, 0, 0, n)) {
    cout << "Solution found!" << endl;

    // Write solution to output file
    ofstream outFile("output_board.txt");
    if (!outFile) {
      cerr << "Error: Cannot create output_board.txt" << endl;
      return 1;
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        outFile << board[i][j];
        if (j < n - 1)
          outFile << " ";
      }
      outFile << endl;
    }
    outFile.close();
  } else {
    cout << "No solution exists!" << endl;
    return 1;
  }

  return 0;
}