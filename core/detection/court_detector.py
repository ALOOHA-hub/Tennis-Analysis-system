import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from utils.logger import logger


class ConvBlock(nn.Module):
    """A single convolutional block: Conv2d -> ReLU -> BatchNorm2d."""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class CourtDetectorNet(nn.Module):
    """TrackNet-style encoder-decoder CNN for tennis court keypoint heatmap prediction."""
    def __init__(self, out_channels=15):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        # Bottleneck
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # Decoder
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x


class CourtDetector:
    """Loads the TrackNet-based court detector and predicts 14 keypoints via heatmaps."""

    # The model expects input at this resolution
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 360
    # Scaling between model output and original video resolution
    SCALE = 2  

    def __init__(self, model_path, device='cpu'):
        logger.info(f"Loading TrackNet Court Detector from {model_path}")
        self.device = device
        self.model = CourtDetectorNet(out_channels=15)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        logger.info("Court Detector loaded successfully.")

    def predict(self, image):
        """Processes a single frame and returns 14 court keypoints dynamically scaled."""
        # 1. Get the actual original video dimensions
        original_h, original_w = image.shape[:2]
        
        # 2. Calculate dynamic scale ratios (Original vs Model Input)
        width_ratio = original_w / self.INPUT_WIDTH
        height_ratio = original_h / self.INPUT_HEIGHT

        # Resize to model input size
        img = cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.0)
        inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            out = self.model(inp.float().to(self.device))[0]
        pred = torch.sigmoid(out).detach().cpu().numpy()

        # Extract keypoints from heatmaps
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            _, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(
                heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                param1=50, param2=2, minRadius=10, maxRadius=25
            )
            if circles is not None:
                # 3. Apply the dynamic ratios instead of the hardcoded SCALE=2
                x_pred = circles[0][0][0] * width_ratio
                y_pred = circles[0][0][1] * height_ratio
                points.extend([x_pred, y_pred])
            else:
                # Use NaN for missing keypoints
                points.extend([np.nan, np.nan])

        return np.array(points)