import cv2
import numpy as np

from src import Camera, FaceDetectorV1, FaceDetectorV2, EmotionDetector
from pipelines import get_camera_pipeline, get_mic_pipeline


def faceDetectPredict(camera, face_detector=None, emotion_detector=None):
	if camera.cap.isOpened():
		cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)

		while True:
			return_key, image = camera.read()
			if not return_key:
				break

			if face_detector:
				faces = face_detector.detect(image)
				for (x1, y1, x2, y2) in faces:
					cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
					if emotion_detector:
						roi = image[y1: y2, x1: x2]
						emotion = emotion_detector.predict(roi)
						cv2.putText(
						image, emotion, (x1, y1), cv2.FONT_HERSHEY_COMPLEX,
						2, (0, 0, 255), 3
						)


			cv2.imshow("Face Detection Window", image)

			key = cv2.waitKey(30) & 0xff
			if key == 27:
				camera.stop()
				cv2.destroyAllWindows()
				break

	else:
		print("Cannot open Camera")


if __name__ == "__main__":

	"""Init & Start camera"""
	cam_pipe = get_camera_pipeline()
	camera = Camera(cam_pipe)
	camera.start()

	face_detector = FaceDetectorV1()
	# emotion_detector = EmotionDetector()

	faceDetectPredict(camera, face_detector, None)

