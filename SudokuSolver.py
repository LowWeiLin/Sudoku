import numpy as np
import cv2
from SudokuExtractor import SudokuExtractor


class SudokuSolver:
    
    potentialSolutions = None

    # puzzle should be a 9x9 array, with 0 representing empty cells,
    # and 1-9 representing fixed values in the puzzle.
    def solve(self, puzzle):
        self.initializeAllPotentialSolutions()

        # Set answers for fixed values in puzzle
        for y in range(0,9):
            for x in range(0,9):
                if puzzle[y][x] != 0:
                    ans = puzzle[y][x]
                    self.setAnswer(x, y, ans)

        potentialAnswersCount = self.countPotentialAnswers()
        while True:
            # run unique contraint check on each cell
            for y in range(0,9):
                for x in range(0,9):
                    self.checkUniqueConstraint(x, y)

            # stop when unique solution is found
            if self.uniqueSolutionFound():
                print "Unique solution found"
                solution = np.array(self.potentialSolutions).reshape((9,9))
                print "Verification result:", self.verify(solution)
                print solution
                return solution
            # stop if no change to potential solutions
            if potentialAnswersCount == self.countPotentialAnswers():
                print "No change. Multiple solutions?"
                break
            # stop if any cell has no potential answers
            if self.noSolutionFound():
                print "No solution found"
                return None

            potentialAnswersCount = self.countPotentialAnswers()

        self.printPotentialSolutions()

        # TODO: Handle case where multiple solutions are possible
        return self.potentialSolutions


    def printPotentialSolutions(self):
        # print potential solutions
        print "Potential solutions:"
        for y in range(0,9):
            print self.potentialSolutions[y]

        # print number of potential answers for each cell
        print "Number of potential answers:"
        for y in range(0,9):
            print map(len,self.potentialSolutions[y])


    checkNumber = (1+2+3+4+5+6+7+8+9)
    def verify(self, puzzle):
        # Check rows
        for y in range(0,9):
            total = sum(puzzle[y,:])
            if total != self.checkNumber:
                return False

        # Check columns
        for x in range(0,9):
            total = sum(puzzle[:,x])
            if total != self.checkNumber:
                return False
                
        # Check boxes
        for y in range(0,9,3):
            for x in range(0,9,3):
                total = sum(map(sum, puzzle[y:y+3,x:x+3]))
                if total != self.checkNumber:
                    return False

        # Check all numbers are between 1-9 inclusive
        for y in range(0,9):
            for x in range(0,9):
                if puzzle[y][x] > 9 or puzzle[y][x] < 1:
                    return False

        return True

    def checkUniqueConstraint(self, x, y):
        # no need to check
        if len(self.potentialSolutions[y][x]) == 1:
            return

        # For each potential answer in given cell
        for i in range(0, len(self.potentialSolutions[y][x])):
            num = self.potentialSolutions[y][x][i]

            # check if potential answer is unique in row
            unique = True
            for _x in range(0, 9):
                if _x != x and (num in self.potentialSolutions[y][_x]):
                    unique = False
                    break
            
            if unique == True:
                self.setAnswer(x, y, num)
                break

            # check if potential answer is unique in col
            unique = True
            for _y in range(0, 9):
                if _y != y and (num in self.potentialSolutions[_y][x]):
                    unique = False
                    break
            
            if unique == True:
                self.setAnswer(x, y, num)
                break

            # check if potential answer is unique in box
            unique = True
            box_x = (x/3) * 3
            box_y = (y/3) * 3
            for _y in range(box_y,box_y+3):
                for _x in range(box_x,box_x+3):
                    if (_x != x or _y != y) and (num in self.potentialSolutions[_y][_x]):
                        unique = False
                        break
                if unique == False:
                    break
            
            if unique == True:
                self.setAnswer(x, y, num)
                break


    def setAnswer(self, x, y, answer):
        self.potentialSolutions[y][x] = [answer]
        # Remove answer from row
        for _x in range(0,9):
            if _x != x and (answer in self.potentialSolutions[y][_x]):
                self.potentialSolutions[y][_x].remove(answer)
                if len(self.potentialSolutions[y][_x]) == 1:
                    self.setAnswer(_x, y, self.potentialSolutions[y][_x][0])

        # Remove answer from col
        for _y in range(0,9):
            if _y != y and (answer in self.potentialSolutions[_y][x]):
                self.potentialSolutions[_y][x].remove(answer)
                if len(self.potentialSolutions[_y][x]) == 1:
                    self.setAnswer(x, _y, self.potentialSolutions[_y][x][0])
                
        # remove answer from box
        box_x = (x/3) * 3
        box_y = (y/3) * 3
        for _y in range(box_y,box_y+3):
            for _x in range(box_x,box_x+3):
                if (_y != y or _x != x) and (answer in self.potentialSolutions[_y][_x]):
                    self.potentialSolutions[_y][_x].remove(answer)
                    if len(self.potentialSolutions[_y][_x]) == 1:
                        self.setAnswer(_x, _y, self.potentialSolutions[_y][_x][0])


    # unique solution found if all cells has only 1 potential answer
    def uniqueSolutionFound(self):
        for y in range(0,9):
            for x in range(0,9):
                if len(self.potentialSolutions[y][x]) != 1:
                    return False
        return True


    # no solution if any cell has 0 potential answers
    def noSolutionFound(self):
        for y in range(0,9):
            for x in range(0,9):
                if len(self.potentialSolutions[y][x]) == 0:
                    return True
        return False

    # count number of potential answers
    def countPotentialAnswers(self):
        total = 0
        for y in range(0,9):
            for x in range(0,9):
                total += len(self.potentialSolutions[y][x])

        return total


    def initializeAllPotentialSolutions(self):
        self.potentialSolutions = []
        for y in range(0,9):
            self.potentialSolutions.append([])
            for x in range(0,9):
                self.potentialSolutions[y].append([])
                for i in range(0,9):
                    self.potentialSolutions[y][x].append(i+1)

        #print self.potentialSolutions

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

#
#   Main Entry Point
#
if __name__ == '__main__':
    extractor = SudokuExtractor()
    extractor.extract("sudoku_original.jpg")
    puzzle = extractor.recognized_puzzle
    solution = SudokuSolver().solve(puzzle)
    extractor.showOverlayPuzzle(solution, "Solved puzzle")
    cv2.waitKey()