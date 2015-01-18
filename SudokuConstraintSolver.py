
import Puzzles
import numpy as np
import cv2
from SudokuExtractor import SudokuExtractor
from SudokuBacktrackingSolver import SudokuBacktrackingSolver


class SudokuConstraintSolver:
    
    potentialSolutions = None

    # puzzle should be a 9x9 array, with 0 representing empty cells,
    # and 1-9 representing fixed values in the puzzle.
    def solve(self, puzzle):
        self.initializeAllPotentialSolutions()

        # Set answers for fixed values in puzzle
        self.setAnswersForFixedValues()

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
            # stop if any cell has no potential answers
            elif self.noSolutionFound():
                print "No solution found"
                self.printPotentialSolutions()
                return []
            # stop if no change to potential solutions
            elif potentialAnswersCount == self.countPotentialAnswers():
                # cannot determmine solution/multiple solutions
                # use backtracking algorithm
                # limit to at most 10 solutions
                return SudokuBacktrackingSolver().solve(puzzle, 10)


            potentialAnswersCount = self.countPotentialAnswers()

        self.printPotentialSolutions()
        
        return self.potentialSolutions


    def setAnswersForFixedValues(self):
        for y in range(0,9):
            for x in range(0,9):
                if puzzle[y][x] != 0:
                    ans = puzzle[y][x]
                    self.setAnswer(x, y, ans)


    def printPotentialSolutions(self):
        # print potential solutions
        print "Potential solutions:"
        for y in range(0,9):
            print self.potentialSolutions[y]

        # print number of potential answers for each cell
        print "Number of potential answers:"
        for y in range(0,9):
            print map(len,self.potentialSolutions[y])


    def verify(self, puzzle):

        # Check all numbers are between 1-9 inclusive
        for y in range(0,9):
            for x in range(0,9):
                if puzzle[y][x] > 9 or puzzle[y][x] < 1:
                    print "Invalid number at (%d, %d): %d" % (y, x, puzzle[y][x])
                    return False

        # Checks that a list contains all of 1-9.
        # Returns the index of an invalid digit (0-8), or -1 if none are invalid.
        def invalid(lst):
            found = {}
            for i, d in enumerate(lst):
                if d in found:
                    return i
                else:
                    found[d] = True
            return -1

        # Check rows
        for y in range(0,9):
            row = puzzle[y,:]
            invalidDigit = invalid(row)
            if invalidDigit >= 0:
                print "Invalid number %d in row %d at position %d" % (row[invalidDigit], y, invalidDigit+1)
                return False

        # Check columns
        for x in range(0,9):
            col = puzzle[:,x]
            invalidDigit = invalid(col)
            if invalidDigit >= 0:
                print "Invalid number %d in column %d at position %d" % (col[invalidDigit], x, invalidDigit+1)
                return False

        # Check boxes
        for y in range(0,9,3):
            for x in range(0,9,3):
                box = puzzle[y:y+3,x:x+3].flatten()
                invalidDigit = invalid(box)
                if invalidDigit >= 0:
                    print "Invalid number %d in box (%d, %d)" % (box[invalidDigit], y/3, x/3)
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


#
#   Main Entry Point
#
if __name__ == '__main__':
    extractor = SudokuExtractor()
    extractor.extract("sudoku_original.jpg")
    puzzle = Puzzles.manySolutions
    solution = SudokuConstraintSolver().solve(puzzle)
    if len(solution) >= 1:
        extractor.showOverlayPuzzle(solution[0], "Solved puzzle")
    cv2.waitKey()
