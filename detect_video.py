import cv2
import numpy as np

from src import Camera, Display, FaceDetectorV1, FaceDetectorV2, EmotionDetector
from pipelines import get_camera_pipeline, get_mic_pipeline


def faceDetectPredict(camera, display, face_detector=None, emotion_detector=None,predict_at_frames=5):
	if camera.cap.isOpened():
		cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)

		frame_count = 0
		faces = []
		texts = []		

		while True:
			return_key, image = camera.read()
			if not return_key:
				break

			frame_count += 1
			if frame_count % predict_at_frames != 0:
				display.show(image, faces, texts)
			else:
				if face_detector:
					faces = face_detector.detect(image)
					for (x1, y1, x2, y2) in faces:
						cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
					if emotion_detector:
						texts = []
						# Note: this predicts each image separately
						for (x1, y1, x2, y2) in faces:
							roi = image[y1: y2, x1: x2]
							emotion = emotion_detector.predict(roi)
							texts.append((emotion, x1, y1))
						# TODO: write some code to process all ROIs all at once
						...

				display.show(image, faces, texts)

			key = cv2.waitKey(30) & 0xff
			if key == 27:
				camera.stop()
				cv2.destroyAllWindows()
				break
	else:
		print("Cannot open Camera")


if __name__ == "__main__":

	# Init & Start camera
	cam_pipe = get_camera_pipeline()
	camera = Camera(cam_pipe)
	camera.start()

	# Init display
	display = Display()

	# Face detector & Emotion classifier
	face_detector = FaceDetectorV1()
	emotion_detector = EmotionDetector()

	faceDetectPredict(camera, display, face_detector, emotion_detector)

