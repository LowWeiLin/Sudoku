
def copyList(xs):
    return [row[:] for row in xs]


class SudokuSolver:
    
    # puzzle should be a 9x9 array, with 0 representing empty cells,
    # and 1-9 representing fixed values in the puzzle.
    # Solves a puzzle represented as a 2D array.

    # The limit parameter controls how many solutions to find
    # (-1 being as many as possible)

    def solve(self, puzzle, limit=-1):

        solutions = []

        def enoughSolutions():
            if limit == -1:
                return False
            else:
                return len(solutions) >= limit

        # Checks if num may be legally placed at cell (x, y)
        def check(x, y, num):
            # Check row
            for xx in range(0, 9):
                if xx != x and puzzle[y][xx] == num:
                    return False

            # Check column
            for yy in range(0, 9):
                if yy != y and puzzle[yy][x] == num:
                    return False

            # Check box
            # First determine the top-left coordinates of the box we're in
            bx = int(x / 3) * 3
            by = int(y / 3) * 3

            # Get contents of the box (in row-major representation)
            box2d = map(lambda a: a[bx:bx+3], puzzle[by:by+3])
            box = sum(box2d, [])

            # Determine this cell's position in the box so we can avoid checking it
            bpx = x % 3
            bpy = y % 3
            posInBox = bpx + bpy * 3

            for i, b in enumerate(box):
                if i != posInBox and b == num:
                    return False

            return True

        def helper(x, y):
            
            if enoughSolutions():
                return

            if y == 9:
                # Making it here means we have a valid answer
                solutions.append(copyList(puzzle))
                return

            # Compute coordinates of next cell (moving right, wrapping on rows)
            nx = (x+1) % 9
            ny = y+1 if x == 8 else y

            # If the current cell is empty, try all possible values for it
            if puzzle[y][x] == 0:
                for n in range(1, 10):
                    puzzle[y][x] = n

                    # Branch from this cell only if consistent, otherwise
                    # prune search tree
                    if check(x, y, n):
                        helper(nx, ny)

                # We've exhausted all values for this cell.
                # Reset it so it'll be in a fresh state when we backtrack and come back to it
                puzzle[y][x] = 0

            # Otherwise proceed to the next cell, not branching
            else:
                helper(nx, ny)

        helper(0, 0)
        return solutions



oneSolution = [
    [0, 0, 8, 6, 0, 2, 0, 0, 0],
    [2, 5, 0, 9, 1, 7, 0, 0, 0],
    [1, 0, 0, 0, 3, 0, 0, 0, 0],
    [6, 4, 0, 0, 0, 0, 5, 0, 0],
    [8, 0, 7, 5, 9, 3, 1, 0, 4],
    [0, 0, 3, 0, 0, 0, 0, 9, 2],
    [0, 0, 0, 0, 6, 0, 0, 0, 8],
    [0, 0, 0, 3, 7, 8, 0, 1, 5],
    [0, 0, 0, 1, 0, 9, 4, 0, 0]
]

manySolutions = [
    [9, 0, 6, 0, 7, 0, 4, 0, 3],
    [0, 0, 0, 4, 0, 0, 2, 0, 0],
    [0, 7, 0, 0, 2, 3, 0, 1, 0],
    [5, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 4, 0, 2, 0, 8, 0, 6, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 5],
    [0, 3, 0, 7, 0, 0, 0, 5, 0],
    [0, 0, 7, 0, 0, 5, 0, 0, 0],
    [4, 0, 5, 0, 1, 0, 7, 0, 8]
]

image3 = [
    [0, 3, 9, 1, 0, 0, 0, 0, 0],
    [4, 0, 8, 0, 6, 0, 0, 0, 2],
    [2, 0, 0, 5, 8, 0, 7, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 9, 0, 0, 0],
    [3, 0, 6, 0, 0, 0, 0, 4, 9],
    [0, 0, 0, 0, 1, 0, 0, 3, 0],
    [0, 4, 0, 3, 0, 0, 0, 0, 8],
    [7, 0, 0, 0, 0, 0, 4, 0, 0]
]

empty = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

#
#   Main Entry Point
#
if __name__ == '__main__':
    # extractor = SudokuExtractor()
    # extractor.extract("sudoku_original.jpg")
    puzzle = manySolutions
    solution = SudokuSolver().solve(puzzle)
    # extractor.showOverlayPuzzle(solution, "Solved puzzle")
    # cv2.waitKey()
