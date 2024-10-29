import numpy as np
import subprocess
import matplotlib.pyplot as plt
import time

from src import Mic, VAD, EmotionDetectorAudio

if __name__ == "__main__":
	
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
	
	samples_ = mic.pre_process_samples(samples)
	print(emo.predict(samples_))

	plt.figure()
	plt.plot(samples_)
	plt.plot(detect * 0.5)
	plt.ylim([-1, 1])
	plt.show()


