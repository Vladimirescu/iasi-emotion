import numpy as np
import time
from datetime import datetime
import threading

from src import Mic, VAD
from pipelines import get_mic_pipeline


def dummy_predict_speech(speech):
    """
    Some dummy function that simulates a "prediction" over speech samples
    """
    
    t = np.random.uniform(0.5, 1)
    time.sleep(t)
    
    return np.random.randint(10)
    

def process_speech_segment(speech_segment, counter):

    pred = dummy_predict_speech(np.array(speech_segment))
    print(f">> Segment #{counter} prediction: {pred}")


def continous_vad_async(rate=48000, duration=0.2, channels=1, stop_threshold=0.8):
    """
    Continuously listens to the microphone, applies VAD, and prints
    'start' with timestamp when speaking begins and 'end' with timestamp when speaking stops.
    Detects a specific stop sound (e.g., a clap) to end the program.
    
    Parameters:
    - rate: Sample rate for the microphone
    - duration: Duration of each segment in seconds (e.g., 0.2 for 200ms)
    - channels: Number of audio channels
    - stop_threshold: Amplitude threshold to detect the "stop sound" (e.g., a clap)
    """
    # Configure the audio pipeline
    pipeline, rate = get_mic_pipeline(duration=duration)
    
    mic = Mic(pipeline)
    vad = VAD(rate)
    
    speaking = False  
    speech_segment = [] 
    segment_counter = 0

    def main_loop():
        nonlocal speaking, speech_segment, segment_counter

        print("Starting continuous VAD detection...")
        while True:
            samples = mic.listen()          
            vad_output = vad.detect(samples) 

            # TODO: some speech data may be lost before the vad starts predicting 1 (also seen in mic_test.py). How can you augument speech_segment s.t. it contains 2-3 frames before vad starts detecting speech?

            if any(vad_output):
                if not speaking:
                    print(f"---Segment #{segment_counter}---")
                    print(f"start - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    speaking = True
                speech_segment.extend(samples)  
            else:
                if speaking:
                    print(f"end - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("----------")

                    threading.Thread(
                        target=process_speech_segment,
                        args=(speech_segment, segment_counter)
                    ).start()
                    
                    segment_counter += 1
                    speech_segment = []  
                    speaking = False

            time.sleep(duration)

    main_thread = threading.Thread(target=main_loop)
    main_thread.start()
    main_thread.join()


if __name__ == "__main__":
    continous_vad_async()
    
