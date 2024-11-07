import yaml
import cv2
from pathlib import Path
import subprocess

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


def get_mic_pipeline(duration=1, file_name=None, jetson=True):
    """Mic pipeline, with specified configuration.
    A duration = 0 will listen to infinity, until process is killed.
    """        
    with open(device_file, "r") as f:
        config = yaml.safe_load(f)
    
    audio = config["audio"]
    
    if jetson:
        audio_pipeline = [
            'ffmpeg', '-y',
            '-f', 'alsa',    
            '-ac', str(audio['channels']),
            '-ar', str(audio['rate']),
            '-i', f"hw:{audio['card']},{audio['device']}",
            # filter for enhancing speech freq range
            # choose the volume according to how sensitive you want the mic to be
            '-af', "highpass=f=10, lowpass=f=3000, volume=2", 
            '-t', str(duration),
            '-f', 
            's16le' if file_name is None else 'wav', 
            'pipe:1' if file_name is None else file_name
        ]
    else:
        """For MAC"""
        audio_pipeline = [
                        'sox', '-d', 
                        '-t', 'raw', 
                        '-b', '16', 
                        '-e', 'signed-integer',
                        '--endian', 'little',
                        '-r', str(rate),           
                        '-c', str(channels),
                        '-', 
                        'trim', '0', str(duration)
                        ]


    return audio_pipeline, audio['rate']


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
    pipe, rate = get_mic_pipeline(duration=2, file_name="test")
    result = subprocess.run(pipe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    test_camera()
    print("Recording..")
    test_audio()
    print("Done")

