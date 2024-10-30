import numpy as np
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import datetime

from src import Mic, VAD, EmotionDetectorAudio


def continuous_vad(rate=48000, duration=0.1, channels=1):
    """
    Continuously listens to the microphone, applies VAD, and prints
    
    Parameters:
    :param rate: sample rate
    :param duration: duration of each segment in seconds (e.g., 0.2 for 200ms)
    :param channels: n.o. of audio channels
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
    
    speaking = False

    print("Starting continuous VAD detection...")
    while True:
        samples = mic.listen()           
        vad_output = vad.detect(samples) 

		print(np.max(samples))

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
	exit()

	"""Config for MAC"""
	rate = 48000
	duration = 2
	channels = 1

	pipeline = [
        'sox', '-d', 
		'-t', 'raw', 
		'-b', '16', 
		'-e', 'signed-integer',
		'--endian', 'little',
        '-r', str(rate),           
        '-c', str(channels),
		'-',
		'trim', '0', str(duration),  
    ]
	"""---------------"""

	mic = Mic(pipeline)
	vad = VAD(rate)
	emo = EmotionDetectorAudio(rate)

	samples = mic.listen()
	t0 = time.time()
	detect = vad.detect(samples)
	t1 = time.time()
	print("VAD delay: ", t1 - t0)
	
	t0 = time.time()
	emotion = emo.predict(samples)
	t1 = time.time()
	print("Emo delay: ", t1 - t0)
	print("Emotion: ", emotion)

	plt.figure()
	plt.plot(mic.pre_process_samples(samples))
	plt.plot(detect * 0.5)
	plt.ylim([-1, 1])
	plt.show()


