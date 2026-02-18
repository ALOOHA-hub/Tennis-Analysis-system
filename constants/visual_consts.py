"""Constants related to the visual annotations (colors, shapes, text)."""

# --- Colors (B, G, R) ---
PLAYER_COLOR = (0, 255, 0)      # Green for players
BALL_COLOR = (0, 255, 255)      # Yellow for the tennis ball
TEXT_COLOR = (0, 0, 0)          # Black text
TEXT_BG_COLOR = (255, 255, 255) # White background for text

# --- Ellipse Settings (Player Feet) ---
ELLIPSE_HEIGHT_RATIO = 0.2
ELLIPSE_START_ANGLE = 45
ELLIPSE_END_ANGLE = 235
ELLIPSE_THICKNESS = 2

# --- Triangle Settings (Ball Marker) ---
TRIANGLE_SIZE = 10
TRIANGLE_Y_OFFSET = 15

# --- Font Settings ---
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# --- Mini-Court Radar Settings ---
RADAR_WIDTH = 800
RADAR_HEIGHT = 1200
RADAR_PADDING = 40
RADAR_BG_COLOR = (45, 125, 45)       # Dark green court background
RADAR_LINE_COLOR = (255, 255, 255)   # White court lines
RADAR_PLAYER_COLOR = (0, 0, 255)     # Red dots for players
RADAR_BALL_COLOR = (0, 255, 255)     # Yellow dot for the ball

# --- Real World Court Dimensions (Meters) ---
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48
SINGLE_LINE_WIDTH = 8.23

# --- Player Approximations (Meters) ---
PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91