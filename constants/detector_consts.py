"""Constants related to the YOLO Detection phase and class mappings."""

# --- Detection Constants ---
DETECTION_BATCH_SIZE = 20
DETECTION_CONFIDENCE_THRESHOLD = 0.25

# --- Class Names (Must match your YOLO model's class names) ---
CLASS_PLAYER = "person"
CLASS_REFEREE = "referee"
CLASS_BALL = "sports ball"      # Change to "ball" if your custom model uses that name
CLASS_GOALKEEPER = "goalkeeper"