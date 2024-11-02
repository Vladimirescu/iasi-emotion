import cv2
import numpy as np
import threading
import time

from src import Camera, Display, FaceDetectorV1, FaceDetectorV2, EmotionDetector
from pipelines import get_camera_pipeline, get_mic_pipeline


"""1st approach - continous capture and prediction"""
def faceDetectPredict(camera, 
                      display, 
                      face_detector=None, 
                      emotion_detector=None,
                      predict_at_frames=3):
    """
    Continuously receives frames and after each predict_at_frames
    performs:
    1) Face detection (if face_detector is given)
    2) Emotion recognition (if emotion_detector is given)
    """
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
                    faces_detections = face_detector.detect(image)

                    if emotion_detector:
                        emotion_detections = []
                        # Note: this predicts each image separately
                        for (x1, y1, x2, y2) in faces_detections:
                            roi = image[y1: y2, x1: x2]
                            # TODO: insteaf of just showing the predicted emotion,
                            # show the names and scores for the top-2 predictions
                            emotion = emotion_detector.predict(roi)
                            emotion_detections.append((emotion, x1, y1))
                            
                        # TODO: write some code to process all ROIs at once
                        # Note: you also need to modify emotion_detector.predict 
                        ...

                    if len(faces_detections) != 0:
                        # This is used just to maintain the predictions drawn onto the image
                        # Reduces "interruptions" between subsequent face detections
                        # Not a very good approach! Other ideas?
                        faces = faces_detections
                        if emotion_detector:
                            texts = emotion_detections

                    t1 = time.time()
                    print(f"Delay: {t1-t0}s")
                display.show(image, faces, texts)

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                camera.stop()
                cv2.destroyAllWindows()
                break
    else:
        print("Cannot open Camera")


if __name__ == "__main__":
    
    detect_emotion = False
    
    # Init & Start camera
    cam_pipe = get_camera_pipeline()
    camera = Camera(cam_pipe)
    camera.start()

    # Init display
    display = Display()

    # Face detector & Emotion classifier
    face_detector = FaceDetectorV1()
    if detect_emotion:
        emotion_detector = EmotionDetector()
    else:
        emotion_detector = None

    faceDetectPredict(camera, display, face_detector, emotion_detector)



