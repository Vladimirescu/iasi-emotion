"""
Here you'll implement a voice-activated emotion detection from camera frames.
In the following, you should:

1. Define your VAD that continuously captures audio frames and detects if there is any speech.
2. Define your face detection function which is activated when speech is detected.
3. Define your emotion recognition function which receives detected faces.
4. Make sure the VAD continues to record and detect speech without being interrupted by video predictor.

The dummy functions presented below are just for exemplification.
"""

def emotion_recognition(faces):
    """
    This function receives an array of faces and performs emotion recognition over each.
    """
    ...
    
    
def face_detection(frame):
    """
    This function receives a camera frame and detects face coordinates.
    """
    ...
    
    
def voice_activity_detection():
    """
    This function continuously captures audio and performs speech detection.
    """
    ...
    
    
if __name__ == "__main__":

    voice_activity_detection()
