"""
This file is used in order to enable the camera input/output.
It is compatible with the Jetson Nano MIPI-CSI camera.
"""

import cv2


class Camera:
	"""Class for handling camera operations."""
	def __init__(self, pipeline):
		self.pipeline = pipeline

	def start(self):
		"""Open the camera and start capturing frames. 

		Raises:
		    RuntimeError: If the camera could not be opened.
		"""
		self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
		if not self.cap.isOpened():
			raise RuntimeError(f"Could not open camera")

	def read(self):
		if self.cap.isOpened():
			return self.cap.read()

	def stop(self):
		"""Stop capturing frames and release the camera."""
		if self.cap is not None:
			self.cap.release()
			self.cap = None
