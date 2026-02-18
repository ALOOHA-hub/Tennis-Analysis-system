"""Constants related to ByteTrack and custom ball interpolation logic."""

# --- Tracker Configuration ---
TRACKER_ACTIVATION_THRESHOLD = 0.25
TRACKER_LOST_BUFFER = 30

# --- Ball Interpolation Constants ---
MAX_PIXEL_MOVE_PER_FRAME = 100  # Max pixels ball can move in 1 frame before deemed a false positive
INTERPOLATE_LIMIT = 20          # Max frames to guess missing ball positions (~0.6s at 30fps)
ROLLING_WINDOW = 5              # Window size for smoothing the ball's trajectory
BFILL_LIMIT = 5                 # Edge padding limit