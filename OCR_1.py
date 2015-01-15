import numpy as np
import cv2

# Summary
# -------
# Digit OCR using kNN
# Modified from http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
# 
# Extracts digits from image by contour
# Crops digit to bounding rectangle
# Resizes and pads image, while maintaining aspect ratio
# Uses pixel values as features for training/recognition
# Uses kNN for classification
#
# Useful Resources
# ----------------
# 1 - http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html
# 2 - http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
#
# Usage
# -----
# Run this python file to generate data file
# 
# Use loadData() to load data file
# Use recognizeCharacter() to recognize a character, by passing in a digit cropped to bounding rectangle.
#

class OCR:
	knn = cv2.KNearest()
	train_data = []
	train_labels = []

	SZ=20
	affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

	def processTrainingImage(self):		
		img = cv2.imread('digits_modified.png')
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		_,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

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
				data = digit_image.reshape(20*20)

				train_data.append(data)
				train_labels.append(label)

		train_data = np.array(train_data).astype(np.float32)
		self.train_data = train_data
		self.train_labels = train_labels

		print "Processed", len(self.train_labels), "characters."


	def saveData(self):
		np.savez('knn_digit_data_1.npz',train_data=self.train_data, train_labels=self.train_labels)


	def loadData(self):
		with np.load('knn_digit_data_1.npz') as data:
			self.train_data = data['train_data']
			self.train_labels = data['train_labels']
		self.knn.train(np.array(self.train_data), np.array(self.train_labels))


	# img takes in image of character cropped to its bounding rectangle
	def recognizeCharacter(self, img):
		feature = self.resizeToSquare(img)
		feature = self.deskew(feature)
		feature = feature.reshape(1, 20*20)
		feature = np.float32(feature)
		ret,result,neighbours,dist = self.knn.find_nearest(feature, k=5)
		return result[0][0]
		

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

if __name__ == '__main__':
	ocr = OCR()
	ocr.processTrainingImage()
	ocr.saveData()
