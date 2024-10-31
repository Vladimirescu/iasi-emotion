import numpy as np
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import datetime

from src import Mic, VAD
from pipelines import get_mic_pipeline


def continuous_vad(rate=48000, duration=0.1, channels=1):
    """
    Continuously listens to the microphone, applies VAD, and prints
    
    Parameters:
    :param rate: sample rate
    :param duration: duration of each segment in seconds (e.g., 0.2 for 200ms)
    :param channels: n.o. of audio channels
    """
    # COnfigure the audio pipeline
    pipeline = get_mic_pipeline(duration=duration)
    
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
		
        time.sleep(duration)  # Wait for the next segment


if __name__ == "__main__":
	
	continuous_vad()



