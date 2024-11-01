import numpy as np
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import datetime
import cv2

from src import Mic, VAD
from pipelines import get_mic_pipeline


def continuous_vad(duration):
    """
    Continuously listens to the microphone, applies VAD, and prints
    
    Parameters:
    :param duration: duration of each segment in seconds to be analyzed (e.g., 0.2 for 200ms)
    :param channels: n.o. of audio channels
    """
    # COnfigure the audio pipeline
    pipeline, rate = get_mic_pipeline(duration=duration)
    
    # Define Mic and VAD objects
    mic = Mic(pipeline)
    vad = VAD(rate)
    
    speaking = False

    print("Starting continuous VAD detection...")
    while True:
        samples = mic.listen()           
        vad_output = vad.detect(samples) 
        
        if any(vad_output):
            if not speaking:
                print(f"------\n start - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                speaking = True
        else:
            if speaking:
                print(f" end - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n------")
                speaking = False
                
        # time.sleep(duration)


if __name__ == "__main__":
    
    continuous_vad(duration=0.2)



