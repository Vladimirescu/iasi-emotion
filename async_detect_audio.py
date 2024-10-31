import numpy as np
import time
import asyncio
from datetime import datetime

from src import Mic, VAD, EmotionDetectorAudio


async def process_speech_segment(speech_segment, emo, counter):

    pred = emo.predict(np.array(speech_segment))
    print("......")
    print(f"Segment #{counter}: {pred}")
    print("......")


def continuous_vad_with_stop_and_processing(rate=48000, duration=0.2, channels=1, stop_threshold=0.8):
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
    pipeline = [
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
    
    mic = Mic(pipeline)
    vad = VAD(rate)
    emo = EmotionDetectorAudio(rate)

    
    speaking = False  
    speech_segment = [] 
    segment_counter = 0

    async def main_loop():
        nonlocal speaking, speech_segment, segment_counter

        print("Starting continuous VAD detection...")
        while True:
            samples = mic.listen()          
            vad_output = vad.detect(samples) 

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

                    asyncio.create_task(
                        process_speech_segment(speech_segment, emo, segment_counter)
                    )
                    
                    segment_counter += 1
                    speech_segment = []  
                    speaking = False

            await asyncio.sleep(duration)  # Non-blocking wait for the next segment

    asyncio.run(main_loop())


if __name__ == "__main__":
    continous_vad()
