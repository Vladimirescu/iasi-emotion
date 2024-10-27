import cv2


class Display:
	def __init__(self):
		self.text_options ={
		    'fontFace': cv2.FONT_HERSHEY_PLAIN,
		    'fontScale': 1,
		    'color': (0, 0, 255),
		    'thickness': 1,
		    'lineType': 2,
		}

	def show(self, image, bboxes=[], texts=[]):
		if isinstance(bboxes, list):
			for box in bboxes:
				x1, y1, x2, y2 = box
				image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

		if isinstance(texts, list):
			for text in texts:
				text_, x, y = text
				image = cv2.putText(image, text_, (x, y), **self.text_options)

		cv2.imshow("Face Detection Window", image)

