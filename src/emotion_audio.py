"""
Emotion recognition from audio
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path

model_file = Path(__file__).parent / "models/emotion_audio/cnn_transf_parallel_model.pt"


class EmotionDetectorAudio():
    def __init__(self, sr):
        """
        Only works for 48kHz audio - hardcoded bottleneck size
        """
        self.labels = {1:'neutral', 
                       2:'calm', 
                       3:'happy', 
                       4:'sad', 
                       5:'angry', 
                       6:'fear', 
                       7:'disgust', 
                       0:'surprise'} 

        self.sr = sr

        assert self.sr == 48_000, ValueError("The network only works for 48kHz audio")

        self.model = ParallelModel(len(self.labels))
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.eval()

        # Signals must be 3s long -> network has layers hardocded tot that dimension
        self.sig_len = self.sr * 3

    def preprocess(self, audio_frames, mx=2**15-1):
        mel_specs = []
        for frame in audio_frames:
            frame = frame.astype(float) / mx

            if len(frame) >= self.sig_len:
                frame = frame[:self.sig_len]
            else:
                offset = np.zeros(int(self.sr * 0.5))
                frame = np.concatenate((offset, frame))

                zeros = np.zeros(self.sig_len - len(frame))
                frame = np.concatenate((frame, zeros))

            mel = librosa.feature.melspectrogram(y=frame,
                                                 sr=self.sr,
                                                 n_fft=1024,
                                                 win_length = 512,
                                                 window='hamming',
                                                 hop_length = 256,
                                                 n_mels=128,
                                                 fmax=self.sr/2
                                                )
            mel = librosa.power_to_db(mel, ref=np.max)

            mel_specs.append(
                torch.tensor(mel, dtype=torch.float32)[None, None, ...]
            )

        # TODO: find out the mean and std of eacj feature from the training data
        mel_specs = torch.cat(mel_specs, 0)
        means = mel_specs.mean(dim=(2, 3), keepdim=True)
        stds = mel_specs.std(dim=(2, 3), keepdim=True)

        mel_specs = (mel_specs - means) / (stds + 1e-7)

        return mel_specs


    def decode_classes(self, classes):
        """
        :param classes: 1D torch.Tensor, containing class indices
        """
        classes = classes.tolist()
        return [self.labels[c] for c in classes]

    def predict(self, audio_frames):
        """
        :param audio_frames: list, individual audio_frames, each in a np.array
        Each audio frame is assumed to be of dtype int16
        """
        if not isinstance(audio_frames, list):
            audio_frames = [audio_frames]

        mels = self.preprocess(audio_frames)
        _, preds = self.model(mels)
        classes = torch.argmax(preds, dim=1)

        return self.decode_classes(classes)


class ParallelModel(nn.Module):
    """
    Audio Emotion classifier.
    Source:
    https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/parallel_cnn_transformer.ipynb
    """
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                       out_channels=16,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(in_channels=16,
                       out_channels=32,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(in_channels=32,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3)
        )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512, dropout=0.4, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        # Linear softmax layer
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x) #(b,channel,freq,time)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension
        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1) 
        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax