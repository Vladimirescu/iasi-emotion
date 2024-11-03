# BIOSINF, IASI Lab 2 - Voice-activated Face detection & Emotion Recognition from Video

In this lab we'll develop a face detection & emotion recognition app, along with a real-time Voice Activity Detection (VAD) to be integrated as a trigger for the video predictor.

## Getting started

Clone this repo onto your local Jetson:
```bash
git clone https://github.com/Vladimirescu/iasi-emotion
```

## Python packages

For this lab, you'll need the following Python packages:
```bash
torch >= 1.9
opencv-python >= 4.0
pyyaml
webrtcvad # for VAD
```

For PyTorch, you'll ned to download the NVIDIA wheel from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048#:~:text=PyTorch-,v1.9.0,-JetPack%204.4%20(L4T)), 
for versions 1.9 or 1.10 supported by Jetpack 4.
To verify which Jetpack version is on your Jetson Nano, run this in CLI:
```bash
apt-cache show nvidia-jetson
```
After downloading the PyTorch wheel, follow [these Installation steps](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048#:~:text=download%20from%20above.-,%3E%20Python%203,-%23%20substitute%20the%20link) to install it and its dependencies.

For all the other packages, simply install them using pip, e.g.:
```bash
pip3 install opencv-python
```

## Test camera & Microphone

To test whether your camera and microphone works, run the following commands:
```bash
cd iasi-emotion
python3 pipelines/get_pipeline.py
```
First, a window showing the camera feed will appear, which can be closed after pressing `Esc`. Afterwards, the audio recording will start for 2 seconds, and a `test.wav` file will
be created in the current folder. You should check if the recorded audio is ok, otherwise you should modify the configuration file `pipelines/device_settings.py` s.t. the correct 
audio device is being used.

## Run video detection scripts

There are 2 scripts for this activity, `detect_video.py` and `async_detect_video.py`. Each of them uses one of the two available face detections, `src/FaceDetectorV1` and `src/FaceDetectorV2`,
and one emotion predictor `src/EmotionDetector`.

There are several `TODO` exercises left for you to complete in `detect_video.py`, and also another in `FaceDetectionV2`.

## Run real-time VAD scripts

Before continuing with real-time VAD app, we'll first look at how this works for a single offline recording. 
For that, we'll run:
```bash
python3 test_mic.py
```
which uses a `src/VAD` object, based on [WebRTC](https://webrtc.org). 
After running, you should see the waveform of your recording, the VAD predictions, and a corresponding spectrogram. 

For the real-time app, we'll use `detect_audio.py` and `async_detect_audio.py` (for exemplification). A `TODO` is left in `detect_audio.py` for you to complete.

## *Exercise*: Combine Video detection with VAD

In this exercise, you'll use the previous functionalities to create a "Real-time" Voice-activated Video Predictor.

You'll have to fill in the script `voice_activ_video.py`, according to the requirements specified in its header.
