import numpy as np
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.signal import spectrogram

from src import Mic, VAD
from pipelines import get_mic_pipeline


def plot_audio_vad(samples, vad_output, rate):
    t = np.linspace(0, duration, samples.shape[0])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t, samples / np.max(samples), label="audio")
    plt.plot(t, vad_output, label="voice activity")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    f, t, Sxx = spectrogram(samples, fs=rate, nfft=1024, nperseg=1024, noverlap=512)
    
    print(Sxx.shape)
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.yscale('log')
    plt.ylim([50, rate / 2])
    
    plt.show()
    


if __name__ == "__main__":
    
    duration = 3 # seconds
    
    # COnfigure the audio pipeline
    pipeline, rate = get_mic_pipeline(duration=duration)
    
    # Define Mic object
    mic = Mic(pipeline)
    vad = VAD(rate)
    
    # Listen and store samples
    print("Recording...")
    samples = mic.listen()
    print("Done.")
    
    vad_output = vad.detect(samples) 
    
    plot_audio_vad(samples, vad_output, rate)
    

    

    
    
    



