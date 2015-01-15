import numpy as np
import cv2
from OCR_1 import OCR

# Summary
# -------
# Recognizes a sudodu puzzle from an image and extracts the puzzle
# *Does not solve the puzzle (yet)
#
# Modified from 
# 1 - http://opencvpython.blogspot.sg/2012/06/sudoku-solver-part-1.html
# 2 - http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-plot/
# Main modifications are in the pre processing of images, and how digits are extracted.
#

class SudokuSolver:
    original_color = None
    original_gray = None
    original_blur_gray = None
    uniform_gray = None
    uniform_blur_gray = None
    threshold_original_blur_gray = None
    threshold_uniform_blur_gray = None

    contour_actual = None
    contour_approx = None

    puzzle_actual_mask = None

    warped_masked_original_color = None
    warped_masked_uniform_gray = None
    warped_masked_uniform_binary = None

    digit_binary = None
    digit_contours = []

    recognized_puzzle = np.zeros(((9,9)), np.uint8)

    ocr = None

    def solve(self):
        self.getPuzzle()
        self.preprocessImages()
        self.findPuzzle()
        self.simpleWarp()
        self.extractDigits()
        self.recognizeDigits()


    def getPuzzle(self):  
        self.original_color = cv2.imread("sudoku_original.jpg")
        cv2.imshow("Original image color", self.original_color)


    def preprocessImages(self):
        self.original_gray = cv2.cvtColor(self.original_color, cv2.COLOR_BGR2GRAY)
        self.uniform_gray = self.normalizeBrightness(self.original_gray)

        self.original_blur_gray = cv2.GaussianBlur(self.original_gray, (5,5), 0)
        self.uniform_blur_gray = cv2.GaussianBlur(self.uniform_gray, (5,5), 0)

        self.threshold_original_blur_gray = cv2.adaptiveThreshold(self.original_blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,2)
        self.threshold_uniform_blur_gray = cv2.adaptiveThreshold(self.uniform_blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,2)


    def normalizeBrightness(self, img_gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        closing_image = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        uniform_image = np.float32(img_gray)/(closing_image)
        uniform_image = np.uint8(cv2.normalize(uniform_image,uniform_image,0,255,cv2.NORM_MINMAX))
        return uniform_image


    def findPuzzle(self):
        # Finds largest contour
        contours, hierarchy = cv2.findContours(self.threshold_original_blur_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_actual = None
        biggest_approx = None
        max_area = 0
        for i in contours:
            # Find contour area
            area = cv2.contourArea(i)
            if area > 100:
                # Get length of perimeter
                peri = cv2.arcLength(i,True)
                # Approximate countour with precision (0.02*peri)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                # Check for largest contour area, with 4 corners
                if area > max_area and len(approx)==4:
                    biggest_approx = approx
                    biggest_actual = i
                    max_area = area

        # Set largest contours
        self.contour_actual = biggest_actual
        self.contour_approx = biggest_approx

        # Get mask
        self.puzzle_actual_mask = np.zeros((self.original_gray.shape),np.uint8)
        cv2.drawContours(self.puzzle_actual_mask,[self.contour_actual],0,255,-1)
        cv2.drawContours(self.puzzle_actual_mask,[self.contour_actual],0,0,2)


    def simpleWarp(self):
        # Get masked uniform image
        masked_uniform_gray = cv2.bitwise_and(self.uniform_gray, self.puzzle_actual_mask)

        # rectify approx
        rectify_contour_approx = self.rectify(self.contour_approx)
        # warp to 450x450 image
        new_coordinates = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
        retval = cv2.getPerspectiveTransform(rectify_contour_approx, new_coordinates)
        warped_masked_uniform_gray = cv2.warpPerspective(masked_uniform_gray, retval, (450,450))
        warped_masked_original_color = cv2.warpPerspective(self.original_color, retval, (450,450))

        _,warped_masked_uniform_binary = cv2.threshold(warped_masked_uniform_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self.warped_masked_uniform_binary = warped_masked_uniform_binary
        self.warped_masked_uniform_gray = warped_masked_uniform_gray
        self.warped_masked_original_color = warped_masked_original_color


    def extractDigits(self):
        warped_masked_uniform_gray_inv = 255-self.warped_masked_uniform_binary

        # Flood fill from 4 corners
        h, w = warped_masked_uniform_gray_inv.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (0,0), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (w-1,0), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (0,h-1), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (w-1,h-1), 0)

        # get contours
        contours, hierarchy = cv2.findContours(warped_masked_uniform_gray_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_contours_overlay_color = self.warped_masked_original_color.copy()
        digit_contours_mask_binary =  np.zeros((warped_masked_uniform_gray_inv.shape),np.uint8)
        digit_binary = None
        digit_contours = []

        for i in contours:
            area = cv2.contourArea(i)
            # Get bounding rectangle
            # height and width of digit should be less than 50
            [_,_,w,h] = cv2.boundingRect(i)
            if area > 50 and w<50 and h<50:
                cv2.drawContours(digit_contours_overlay_color, [i], 0, (0,255,0), 2)
                cv2.drawContours(digit_contours_overlay_color, [i], 0, (0,255,0), -1)

                cv2.drawContours(digit_contours_mask_binary, [i], 0, 255, 2)
                cv2.drawContours(digit_contours_mask_binary, [i], 0, 255, -1)

                digit_contours.append(i)

        digit_binary = cv2.bitwise_and(digit_contours_mask_binary, warped_masked_uniform_gray_inv) * 255

        self.digit_binary = digit_binary
        self.digit_contours = digit_contours


    def recognizeDigits(self):
        # Initialize OCR if it is not yet initialized
        if self.ocr == None:
            self.ocr = OCR()
            self.ocr.loadData()

        digit_binary = self.digit_binary.copy()
        # Pad borders so that digits close to edges can still be extracted
        padding = 50
        digit_binary = cv2.copyMakeBorder(digit_binary,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)

        # Process and recognize each digit
        for i in self.digit_contours:
            # Get centroid
            M = cv2.moments(i)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Find which cell it belongs to
            cell = (int(cx/50), int(cy/50))

            # Get bounding rectangle
            [x,y,w,h] = cv2.boundingRect(i)

            # Extact digit
            # Get image of the digit, cropped o bounding rectangle
            # Add padding to account for padding of digit_binary
            digit = digit_binary[y+padding:y+h+padding,x+padding:x+w+padding]

            # Skip if unable to get image
            if len(digit) == 0:
                continue

            # Set results
            self.recognized_puzzle[cell[1]][cell[0]] = self.ocr.recognizeCharacter(digit)

        # Draw text on image to verify
        warped_masked_original_color = self.warped_masked_original_color.copy()
        for y in range(0,9):
            for x in range(0,9):
                num = self.recognized_puzzle[y][x]
                if num != 0:
                    cv2.putText(warped_masked_original_color,str(num), (x*50+25,y*50+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        cv2.imshow("Recognized puzzle", warped_masked_original_color)


    # Rectify - reshapes and sorts the order of points in a contour.
    def rectify(self, h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
  
        return hnew


#
#   Main Entry Point
#
if __name__ == '__main__':
    SudokuSolver().solve()
    cv2.waitKey()
