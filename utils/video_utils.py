import cv2
import numpy as np
from typing import List
from core.utils.logger import logger

def read_video(video_path: str) -> List[np.ndarray]:
    """Reads a video and returns a list of frames."""
    logger.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    logger.info(f"Successfully read {len(frames)} frames from {video_path}")
    return frames

def save_video(output_video_frames: List[np.ndarray], output_video_path: str, fps: float = 24.0):
    """Saves a list of frames to a video file."""
    if not output_video_frames:
        logger.error("No frames provided to save. Aborting.")
        raise ValueError("No frames provided to save.")
        
    logger.info(f"Initializing video writer for: {output_video_path} at {fps} FPS")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    height, width, _ = output_video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(output_video_frames):
        out.write(frame)
        
    out.release()
    logger.info(f"Successfully saved video to {output_video_path}")