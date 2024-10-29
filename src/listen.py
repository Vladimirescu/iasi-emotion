"""
This file is intended to enable the microphone.
"""
import subprocess
import numpy as np
import webrtcvad


class Mic():
    def __init__(self, pipeline):
        """
        :param pipeline: list, command to record audio 
        """

        self.pipeline = pipeline
    
    @staticmethod
    def pre_process_samples(samples, max_val=2**15-1):
        
        return samples.astype(float) / max_val

    def listen(self):
        result = subprocess.run(self.pipeline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if result.returncode != 0:
            print(result.stderr.decode("utf-8"))
            exit()

        samples = np.frombuffer(result.stdout, dtype=np.int16).squeeze()

        return samples


class VAD():
    """
    Voice Activity Detector
    """
    def __init__(self, rate, frame_detect_size=30):
        """
        :param rate: sample rate of audio
        :param frame_detect_size: float, size of analysis frame, in miliseconds
        """
        # accepts 16-bit mono PCM audio, sampled at 8000, 16000, or 32000 Hz
        self.vad = webrtcvad.Vad(3)
        self.rate = rate
        self.frame_size = int(rate * frame_detect_size / 1000)

    def detect(self, samples):
        """
        :param samples: 1D np.array containing audio samples
        Returns:
            is_audio: 1D np.array
        """

        num_frames = len(samples) // self.frame_size

        # Create an output array for speech detection
        vad_output = np.zeros(len(samples), dtype=int)

        for frame_index in range(num_frames):
            start = frame_index * self.frame_size
            stop = start + self.frame_size
            frame = samples[start: stop]

            # Check if the frame contains speech
            if self.vad.is_speech(frame.tobytes(), self.rate):
                vad_output[start: stop] = 1  # Mark this frame as speech

        return vad_output