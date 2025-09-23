import os
import json 

# The name of the app, to check if it's running
TARGET_APP_PKG = "com.alibaba.aliexpresshd/com.aliexpress.module.webview"

# for debugging purposes, set to True to enable image saving
IMAGE_DEBUG = False

# If you run the script directly on mobile, set this to True to disable
# incompatible functions, like real-time image view, and configure for this
RUN_ON_MOBILE = False

# If you want to check the energy level, you need Tesseract installed and configured
# When the energy level is under MIN_ENERGY_LEVEL, the game will exit
CHECK_ENERGY_LEVEL = True

# Auto-press the generator when no match is found, only if check energy level is enabled
if CHECK_ENERGY_LEVEL:
    # When there's no match, generate objects from the generators
    # Minimum energy to generate items
    MIN_ENERGY_LEVEL = 2

# Get the energy from the 15 seconds product list view
AUTO_FARM_ENERGY = True and CHECK_ENERGY_LEVEL

# Only try to get energy 10 times
MAX_FARM_SESSIONS = 10

# The first N squares will be ignored. Adjust to your number of e.g., generators.
IGNORED_MATCH_POSITIONS = 9

# If your ignored positions aren't in order then you can add selected ones into a list
ADDITIONAL_IGNORED_POSITIONS = [1,14]

# Define the similarity threshold between items
SIMILARITY_THRESHOLD = 0.85

# If there are a maximum of X matches groups left, press the generators
# You can set to 0 if you want to use the generator when there's no match
MAX_GENERATOR_GROUP_NUMBERS = 1

# Try to automatically delivery the items
AUTOMATIC_DELIVERY = True

# NOTE: These values use relative coordinates (0.0-1.0 range) and work across different resolutions.
# They represent percentages of height or width, making them resolution-independent.
ROI_TOP = 0.497
ROI_BOTTOM = 0.802
ROI_PADDING = 0.029
# Energy left number position
ENG_TOP = 0.064
ENG_BOTTOM = 0.084
ENG_LEFT = 0.622
ENG_RIGHT = 0.704
# Energy browse deals "Go" button position
GO_TOP = 0.62
GO_LEFT = 0.276
# Exit "X" button from task list position
EX_TOP = 0.15
EX_LEFT = 0.922
# Delivery require list position
DEL_TOP = 0.318
# Delivery button position
DEL_BTN_TOP = 0.354
# Delivery button spacing from right to left (relative to width)
DEL_BTN_SPACING = 0.338
# Padding right of the delivery buttons (relative to width)
DEL_BTN_PADDING_RIGHT = 0.116
# Space between grid squares (relative to min(width, height))
GRID_PADDING = 0.003
# Right generator button position
R_GEN_TOP = 0.848
R_GEN_LEFT = 0.734
# Left generator button position
L_GEN_TOP = 0.846
L_GEN_LEFT = 0.264

# Minimum blank spaces on the grid for bot to run
MIN_SPACES_ON_BOARD = 2

# Check if config file exists and load the parameters
config_path = os.path.join(os.getcwd(), "bot_config.json")

if os.path.exists(config_path):
    with open(config_path, "r") as json_file:
        loaded_data = json.load(json_file)

    RUN_ON_MOBILE = loaded_data["RUN_ON_MOBILE"]
    IGNORED_MATCH_POSITIONS = loaded_data["IGNORED_MATCH_POSITIONS"]
    ADDITIONAL_IGNORED_POSITIONS = loaded_data["ADDITIONAL_IGNORED_POSITIONS"]
    ROI_TOP = loaded_data["ROI_TOP"]
    ROI_BOTTOM = loaded_data["ROI_BOTTOM"]
    ROI_PADDING = loaded_data["ROI_PADDING"]
    ENG_TOP = loaded_data["ENG_TOP"]
    ENG_BOTTOM = loaded_data["ENG_BOTTOM"]
    ENG_LEFT = loaded_data["ENG_LEFT"]
    ENG_RIGHT = loaded_data["ENG_RIGHT"]
    GO_TOP = loaded_data["GO_TOP"]
    GO_LEFT = loaded_data["GO_LEFT"]
    EX_TOP = loaded_data["EX_TOP"]
    EX_LEFT = loaded_data["EX_LEFT"]
    DEL_TOP = loaded_data["DEL_TOP"]
    DEL_BTN_TOP = loaded_data["DEL_BTN_TOP"]
    GRID_PADDING = loaded_data["GRID_PADDING"]
    MIN_ENERGY_LEVEL = loaded_data["MIN_ENERGY_LEVEL"]
    MAX_FARM_SESSIONS = loaded_data["MAX_FARM_SESSIONS"]
    SIMILARITY_THRESHOLD = loaded_data["SIMILARITY_THRESHOLD"]
    MAX_GENERATOR_GROUP_NUMBERS = loaded_data["MAX_GENERATOR_GROUP_NUMBERS"]
    MIN_SPACES_ON_BOARD = loaded_data["MIN_SPACES_ON_BOARD"]
    DEL_BTN_SPACING = loaded_data["DEL_BTN_SPACING"]
    DEL_BTN_PADDING_RIGHT = loaded_data["DEL_BTN_PADDING_RIGHT"]
    R_GEN_TOP = loaded_data["R_GEN_TOP"]
    R_GEN_LEFT = loaded_data["R_GEN_LEFT"]
    L_GEN_TOP = loaded_data["L_GEN_TOP"]
    L_GEN_LEFT = loaded_data["L_GEN_LEFT"]
else:
    print(f"The file {config_path} does not exist. Using default values.")