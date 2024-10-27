import cv2
import numpy as np
import threading
import time

from src import Camera, Display, FaceDetectorV1, FaceDetectorV2, EmotionDetector
from pipelines import get_camera_pipeline, get_mic_pipeline


"""1st approach - continous capture and prediction"""
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
					t0 = time.time()
					faces = face_detector.detect(image)
					t1 = time.time()
					print(f"Face detection: {t1-t0}")

					if emotion_detector:
						texts = []
						# Note: this predicts each image separately
						for (x1, y1, x2, y2) in faces:
							roi = image[y1: y2, x1: x2]
							emotion = emotion_detector.predict(roi)
							texts.append((emotion, x1, y1))
						# TODO: write some code to process all ROIs all at once
						# Note: you also need to modify emotion_detector.predict 
						...

				display.show(image, faces, texts)

			key = cv2.waitKey(30) & 0xff
			if key == 27:
				camera.stop()
				cv2.destroyAllWindows()
				break
	else:
		print("Cannot open Camera")


"""2nd approach - Threading"""
last_frame = None
bboxes = None
texts = None
frame_lock= threading.Lock()

def camVideo(camera, display):
	global last_frame
	global bboxes
	global texts
	while camera.cap.isOpened():
		cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)
		return_key, image = camera.read()
		
		with frame_lock:
			bboxes_ = bboxes
			texts_ = texts
			last_frame = image

		display.show(image, bboxes_, texts_)

		key = cv2.waitKey(30) & 0xff
		if key == 27:
			camera.stop()
			cv2.destroyAllWindows()
	return


def predictFrame(camera, face_detector, emotion_detector):
	global last_frame
	global bboxes
	global texts
	while True:
		if not camera.cap.isOpened():
			break

		with frame_lock:
			image = last_frame.copy() if last_frame is not None else None
		if image is not None:
			if face_detector:
				bboxes_ = face_detector.detect(image)
				
				if emotion_detector:
					texts_ = []
					# Note: this predicts each image separately
					for (x1, y1, x2, y2) in bboxes_:
						roi = image[y1: y2, x1: x2]
						emotion = emotion_detector.predict(roi)
						texts_.append((emotion, x1, y1))
					# TODO: write some code to process all ROIs all at once
					# Note: you also need to modify emotion_detector.predict 
					...

					with frame_lock:
						bboxes = bboxes_
						texts = texts_
				else:
					with frame_lock:
						bboxes = bboxes_
				
	return



if __name__ == "__main__":

	approach = 2
	
	# Init & Start camera
	cam_pipe = get_camera_pipeline()
	camera = Camera(cam_pipe)
	camera.start()

	# Init display
	display = Display()

	# Face detector & Emotion classifier
	face_detector = FaceDetectorV1()
	emotion_detector = EmotionDetector()

	if approach == 1:
		### 1st approach
		faceDetectPredict(camera, display, face_detector, emotion_detector)
	else:
		### 2nd approach
		video_thread = threading.Thread(target=camVideo, args=(camera, display))
		predict_thread = threading.Thread(target=predictFrame, args=(camera, face_detector, emotion_detector))

		video_thread.start()
		time.sleep(1)
		predict_thread.start()
		
		video_thread.join()
		predict_thread.join()
	



