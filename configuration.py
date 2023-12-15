import os
import json 

# The name of the app, to check if it's running
TARGET_APP_PKG = "com.alibaba.aliexpresshd/com.aliexpress.module.webview"

# If you run the script directly on mobile, set this to True to disable
# incompatible functions, like real-time image view, and configure for this
RUN_ON_MOBILE = False

# If you want to check the energy level, you need Tesseract installed and configured
# When the energy level is under 2, the game will exit
CHECK_ENERGY_LEVEL = True

# Auto-press the generator when no match is found, only if check energy level is enabled
if CHECK_ENERGY_LEVEL:
    # Generator positions to press, in a list
    GENERATOR_POSITIONS = [1, 2, 3, 4]
    # When there's no match, generate objects from each of these generators
    # Minimum energy to generate items
    MIN_ENERGY_LEVEL = 3

# Get the energy from the 15 seconds product list view
AUTO_FARM_ENERGY = True and CHECK_ENERGY_LEVEL

# Only try to get energy 3 times
MAX_FARM_SESSIONS = 3

# The first 10 squares will be ignored. Adjust to your number of e.g., generators.
IGNORED_MATCH_POSITIONS = 9

# Define the similarity threshold between items
SIMILARITY_THRESHOLD = 0.85

# If there are a maximum of X matches groups left, press the generators
# You can set to 0 if you want to use the generator when there's no match
MAX_GENERATOR_GROUP_NUMBERS = 1

# Try to automatically delivery the items
AUTOMATIC_DELIVERY = True

# NOTE: you should adjust these based on your phone display resolution.
# These are for 1080x2400 and represent percentages of height or width.
ROI_TOP = 0.355  # 852px  height
ROI_BOTTOM = 0.9025  # 2166px height
ROI_PADDING = 0.0287  # 31px  width
# Energy left number position
ENG_TOP = 0.05  # 120px  height
ENG_BOTTOM = 0.07  # 168px  height
ENG_LEFT = 0.484  # 523px  width
ENG_RIGHT = 0.566  # 612px  width
# Energy browse deals "Go" button position
GO_TOP = 0.6065  # 1455px height
GO_LEFT = 0.276  # 298px  width
# Exit "X" button from task list position
EX_TOP = 0.145  # 350px  height
EX_LEFT = 0.926  # 1000px width
# Delivery require list position
DEL_TOP = 0.190
# Delivery btn position
DEL_BTN_TOP = 0.240
# Space between grid squares, px
GRID_PADDING = 7

# Check if config file exists file exists and load the parameters
config_path = os.path.join(os.getcwd(), "bot_config.json")

if os.path.exists(config_path):
    with open(config_path, "r") as json_file:
        loaded_data = json.load(json_file)

    RUN_ON_MOBILE = loaded_data["RUN_ON_MOBILE"]
    IGNORED_MATCH_POSITIONS = loaded_data["IGNORED_MATCH_POSITIONS"]
    GENERATOR_POSITIONS = loaded_data["GENERATOR_POSITIONS"]
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
else:
    print(f"The file {config_path} does not exist. Using default values.")