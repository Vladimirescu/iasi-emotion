import yaml
import cv2
from pathlib import Path

device_file = Path(__file__).parent / "device_settings.yaml"


def get_camera_pipeline():
	"""Camera pipeline, with specified configuration."""
	with open(device_file, "r") as f:
		config = yaml.safe_load(f)
	
	cam = config["cam"]

	camera_pipeline = [
			   "nvarguscamerasrc !", 
		           "video/x-raw(memory:NVMM),", 
			   f"width={cam['width']},", 
			   f"height={cam['height']},", 
			   "format=(string)NV12,", 
			   f"framerate={cam['fps']}/1 !",
			   f"nvvidconv flip-method={cam['flip']} !",
			   "video/x-raw,", 
			   "format=(string)BGRx !", 
			   "videoconvert !",
			   "video/x-raw,", 
			   "width=1200, height=660,",
			   "format=(string)BGR !",
			   "appsink"
			  ]

	return " ".join(camera_pipeline)


def get_mic_pipeline(duration=1, file_name="test.mp3"):
	"""Mic pipeline, with specified configuration."""
	with open(device_file, "r") as f:
		config = yaml.safe_load(f)
	audio = config["audio"] 

	audio_pipeline = [
			    'arecord', 
			    '-D', f"hw:{audio['card']},{audio['device']}", 
			    '-c', str(audio['channels']), 
			    '-r', str(audio['rate']), 
			    '-f', audio['format'], 
			    '-d', str(duration),
			    '-t', 'raw'
			 ]

	return audio_pipeline


def test_camera():
    video_capture = cv2.VideoCapture(get_camera_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)

        while True:
            return_key, image = video_capture.read()
            if not return_key:
                break

            cv2.imshow("Face Detection Window", image)

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")


def test_audio():
	pass


if __name__ == "__main__":
	test_camera()

