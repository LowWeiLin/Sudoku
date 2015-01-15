import numpy as np
import cv2

# Summary
# -------
# Digit OCR using SVM
# Modified from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
# 
# Extracts digits from image by contour
# Crops digit to bounding rectangle
# Resizes and pads image, while maintaining aspect ratio
# Deskews image
# Uses Histogram of Oriented Gradients as features
# Uses SVM for classification
#
# Useful Resources
# ----------------
# 1- http://docs.opencv.org/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
# 
# Usage
# -----
# Run this python file to generate data file
# 
# Use loadData() to load data file
# Use recognizeCharacter() to recognize a character, by passing in a digit cropped to bounding rectangle.
#

class OCR:
	SZ=20
	bin_n = 16 # Number of bins

	svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	                    svm_type = cv2.SVM_C_SVC,
	                    C=2.67, gamma=5.383 )

	affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

	svm = cv2.SVM()
	train_data = []
	train_labels = []

	def processTrainingImage(self):		
		img = cv2.imread('digits_modified.png')
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#_,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

		train_data = []
		train_labels = []
		
		# get contours
		contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for i in contours:
		    area = cv2.contourArea(i)
		    [x,y,w,h] = cv2.boundingRect(i)
		    if area>18:
				label = int(y/100)

				digit_image = gray[y:y+h,x:x+w]
				digit_image = self.resizeToSquare(digit_image)
				digit_image = self.deskew(digit_image)
				data = self.hog(digit_image)

				train_data.append(data)
				train_labels.append([label])

		train_data = np.float32(train_data).reshape(-1,64)
		train_labels = np.float32(train_labels)

		self.train_data = train_data
		self.train_labels = train_labels

		self.svm.train(self.train_data, self.train_labels, params=self.svm_params)
		print "Processed", len(self.train_labels), "characters."


	def saveData(self):
		self.svm.save('svm_digit_data_2.dat')


	def loadData(self):
		self.svm.load('svm_digit_data_2.dat')


	# img takes in image of character cropped to its bounding rectangle
	def recognizeCharacter(self, img):
		feature = self.resizeToSquare(img)
		feature = self.deskew(feature)
		feature = self.hog(feature)
		feature = np.float32(feature)
		result = self.svm.predict(feature)
		return result
		

	# Resize image to a square. Maintains scale and pads edges if necessary.
	def resizeToSquare(self, image, length=16, padding=2):
		(h, w) = image.shape
		ratio = min(float(length)/float(w), float(length)/float(h))
		image_resized = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
		(h, w) = image_resized.shape
		if (h, w) != (length, length):
			image_resized = cv2.copyMakeBorder(image_resized, (length-h)/2, 0, (length-w)/2, 0, cv2.BORDER_CONSTANT, value=0)
		(h, w) = image_resized.shape
		if (h, w) != (length, length):
			image_resized = cv2.copyMakeBorder(image_resized, 0, length-h, 0, length-w, cv2.BORDER_CONSTANT, value=0)
		image_resized = cv2.copyMakeBorder(image_resized, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)	
		return image_resized


	def deskew(self, img):
		m = cv2.moments(img)
		if abs(m['mu02']) < 1e-2:
			return img.copy()
		skew = m['mu11']/m['mu02']
		M = np.float32([[1, skew, -0.5*self.SZ*skew], [0, 1, 0]])
		img = cv2.warpAffine(img,M,(self.SZ, self.SZ),flags=self.affine_flags)
		return img

	def hog(self, img):
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)

		# quantizing binvalues in (0...16)
		bins = np.int32(self.bin_n*ang/(2*np.pi))

		# Divide to 4 sub-squares
		bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), self.bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		return hist


if __name__ == '__main__':
	ocr = OCR()
	ocr.processTrainingImage()
	ocr.saveData()
