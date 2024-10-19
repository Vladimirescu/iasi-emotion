"""
Face detector class
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

model_file = Path(__file__).parent / "models/emotion/emotion_detection_model_state.pth"


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(6), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


class EmotionDetector:
	"""Class for handling emotion detection from ROIs."""
	def __init__(self):
		self.class_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprise"]
		self.model = ResNet(1, len(self.class_labels))
		model_state = torch.load(str(model_file))
		self.model.load_state_dict(model_state)

	def predict(self, img):
		"""
		Predict emotion on grayscale images.

		:param img: numpy.ndarray of shape (H, W, C)
		Returns:
		prediction: string
		"""
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
		img = torch.Tensor(img, dtype=float).unsqueeze(0).unsqueeze(0) / 255.0

		pred = self.model(img)
		pred = torch.max(pred, dim=1)[1].tolist()
		label = self.class_labels[pred[0]]
		
		return label

















