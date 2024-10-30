"""
Face detector class
"""
import cv2
import numpy as np
from pathlib import Path

proto_txt_file = Path(__file__).parent / "models/face/deploy.prototxt.txt"
model_file = Path(__file__).parent / "models/face/res10_300x300_ssd_iter_140000.caffemodel"
haar_file = Path(__file__).parent / "models/face/haarcascade_frontalface_default.xml"


class FaceDetectorV1:
	"""Class for handling Face Detection module."""
	def __init__(self, min_confidence=0.75):
		self.net = cv2.dnn.readNetFromCaffe(
			str(proto_txt_file), str(model_file)
		)
		self.preprocess = lambda x: cv2.dnn.blobFromImage(
			cv2.resize(x, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
		)

		self.min_confidence = min_confidence

	def detect(self, img):
		"""
		Predict face bounding-boxes.

		:param img: numpy.ndarray of shape (H, W, C)
		Returns:
		detections: list of tuples (x1, y1, x2, y2)
		"""
		h, w, _ = img.shape		

		blob = self.preprocess(img)
		self.net.setInput(blob)
		detections = self.net.forward()
		boxes = []
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > self.min_confidence:
				# compute the (x, y)-coordinates of the bounding box on original coords
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
		 
				boxes.append((startX, startY, endX, endY))

		return boxes


class FaceDetectorV2:
	"""Class for handling Face Detection module."""
	def __init__(self):
		# This only detects faces frontal to the camera!
		self.face_cascade = cv2.CascadeClassifier(str(haar_file))

		# Add another classifier for lateral faces images
		# TODO: create an additional classifier 
		...

	def detect(self, img):
		"""
		Predict face bounding-boxes.

		:param img: numpy.ndarray of shape (H, W, C)
		Returns:
		detections: list of tuples (x1, y1, x2, y2)
		"""
		
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		detected_faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
		boxes = []
		for (x_pos, y_pos, width, height) in detected_faces:
			boxes.append((x_pos, y_pos, x_pos + width, y_pos + width))			

		# TODO: augument these frontal predicted boxes with lateral ones
		# Note: you should filter out predictions overlapped "too much" with others		
		...

		return boxes

