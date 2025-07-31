# SUDOKU BUDDY

A smart Sudoku solver that combines the power of **OpenCV + CNN + C++** backtracking to recognize, read, and solve Sudoku puzzles straight from an image. It is using **flask backend** to do so.

## Implementation Overview

So the idea is simple, OpenCV extracts the largest area contour from the image uploaded by the user. Then I applied certain WARP perspective to have a bird eye view and used the CNN model (currently a basic model trained on 10K+ images of digits). The fetched digits are thrown into the C++ solver file. It uses a basic backtracking solution to solve the grid and store the output in form of output_board.txt. The missing digits are finally overlayed on the original image of unsolved sudoku grid. And hence our solution😎😎
