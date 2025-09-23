import platform
import os

def detect_termux():
    system = platform.system()
    
    if system == "Windows":
        return False
    
    elif system == "Linux":
        if "ANDROID_STORAGE" in os.environ or "com.termux" in os.environ.get("HOME", ""):
            return True
        else:
            return False

    else:
        raise ValueError(f"Unknown platform: {system}")

import time
import io
import cv2
import numpy as np
import json
import shutil
import configuration as c

# Import based on configuration
run_on_mobile = detect_termux()
if run_on_mobile:
    import termuxgui as tg
else:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

SCREENSHOT_NAME = "screenshot.jpg"
SCREENSHOT_ANNON = "screenshot_annon.jpg"

# Global variables for GUI elements
root = None
image_label = None
right_frame = None
ignored_matches = None
ignored_matches_add = None
setting_entries = {}

def annotate_image(img):
    height, width, _ = img.shape
    roi_min, roi_max, width_padding = (
        int(c.ROI_TOP * height),
        int(c.ROI_BOTTOM * height),
        int(width * c.ROI_PADDING),
    )
    eng_top, eng_bot, eng_left, eng_right = (
        int(c.ENG_TOP * height),
        int(c.ENG_BOTTOM * height),
        int(c.ENG_LEFT * width),
        int(c.ENG_RIGHT * width),
    )
    go_btn_top, go_btn_left = int(c.GO_TOP * height), int(c.GO_LEFT * width)
    close_btn_top, close_btn_left = int(c.EX_TOP * height), int(c.EX_LEFT * width)
    delivery_top = int(c.DEL_TOP * height)
    delivery_btn_top = int(c.DEL_BTN_TOP * height)
    r_gen_btn_top, r_gen_btn_left = (int(c.R_GEN_TOP * height), int(c.R_GEN_LEFT * width))
    l_gen_btn_top, l_gen_btn_left = (int(c.L_GEN_TOP * height), int(c.L_GEN_LEFT * width))

    max_rows, max_col = 5, 7
    padding = int(c.GRID_PADDING * min(width, height))  # Convert to absolute pixels
    square_size = (roi_max - roi_min) // max_rows
    contours = []

    for row in range(max_rows):
        for col in range(max_col):
            x = (col * square_size) + int(width_padding) + padding
            y = roi_min + row * square_size + padding
            contour = np.array([
                (x, y),
                (x + square_size - 2 * padding, y),
                (x + square_size - 2 * padding, y + square_size - 2 * padding),
                (x, y + square_size - 2 * padding),
            ])
            contours.append(contour)

    # Mark ROI on image
    cv2.rectangle(img, (width_padding, roi_min), (width - width_padding, roi_max), (0, 255, 255), 6)
    
    # Mark energy position, X and "GO" button
    cv2.rectangle(img, (eng_left, eng_top), (eng_right, eng_bot), (255, 0, 255), 6)
    img = cv2.circle(img, (go_btn_left, go_btn_top), 20, (0, 255, 0), -1)
    img = cv2.circle(img, (close_btn_left, close_btn_top), 20, (0, 0, 255), -1)
    img = cv2.circle(img, (r_gen_btn_left, r_gen_btn_top), 25, (0, 255, 255), -1)
    img = cv2.circle(img, (l_gen_btn_left, l_gen_btn_top), 25, (255, 255, 0), -1)

    # Draw line and points for automatic delivery - Convert to relative coordinates
    img = cv2.line(img, (width // 2, delivery_top), (width, delivery_top), (255, 255, 0), 10)
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width * 2), delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width), delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width), delivery_btn_top), 20, (255, 50, 255), -1)

    # Draw grid
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 2)

    # Draw ignored contours (only if IGNORED_MATCH_POSITIONS > 0)
    if c.IGNORED_MATCH_POSITIONS > 0:
        for ig in range(c.IGNORED_MATCH_POSITIONS):
            cv2.drawContours(img, [contours[ig]], 0, (0, 0, 255), 4)

    # Draw additional ignored contours
    for ig in c.ADDITIONAL_IGNORED_POSITIONS:
        if ig > 0 and ig <= len(contours):  # Ensure valid position
            cv2.drawContours(img, [contours[ig - 1]], 0, (0, 0, 255), 4)

    return img

# Create a Termux:GUI button with specified text and layout
def create_button(activity, text, layout, width=0):
    button = tg.Button(activity, text, layout)
    button.settextsize(16)
    button.setlinearlayoutparams(1)
    if width:
        button.setwidth(width)
    return button

# Annotate an image and load it into the image viewer for mobile display
def annotate_and_load_img(screenshot_path, image_viewer):
    img = cv2.imread(screenshot_path)
    annotated_img = annotate_image(img)
    screenshot_anon_path = os.path.expanduser(f"~/{SCREENSHOT_ANNON}")
    cv2.imwrite(screenshot_anon_path, annotated_img)
    with io.open(screenshot_anon_path, "rb") as f:
        image = f.read()
        image_viewer.setimage(image)

# Adjust a configuration attribute by a given value
def adjust_config(attr, value):
    current = getattr(c, attr)
    setattr(c, attr, current + value)
    if not run_on_mobile:
        update_image()

# Adjust energy position vertically (both top and bottom)
def adjust_energy_position(value):
    c.ENG_TOP += value
    c.ENG_BOTTOM += value
    if not run_on_mobile:
        update_image()

# Adjust energy position horizontally (both left and right)
def adjust_energy_horizontal(value):
    c.ENG_LEFT += value
    c.ENG_RIGHT += value
    if not run_on_mobile:
        update_image()

# Open file dialog to select screenshot or use termux-storage-get on mobile
def pick_screenshot():
    if run_on_mobile:
        screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
        os.system(f"termux-storage-get {screenshot_path}")
    else:
        filename = filedialog.askopenfilename(
            title="Select screenshot",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            # Copy to working directory
            screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
            shutil.copy(filename, screenshot_path)
            load_screenshot()

# Load and display the screenshot in the GUI
def load_screenshot():
    screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
    if os.path.exists(screenshot_path):
        if not run_on_mobile:
            update_image()

# Update the displayed image with current annotations
def update_image():
    global image_label, right_frame
    if not run_on_mobile and image_label:
        screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
        if os.path.exists(screenshot_path):
            img = cv2.imread(screenshot_path)
            annotated_img = annotate_image(img)
            
            # Convert to PIL Image for tkinter
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Get current available space in the right frame
            try:
                # Update the frame to get current dimensions
                right_frame.update_idletasks()
                available_width = right_frame.winfo_width()  # Account for padding
                available_height = right_frame.winfo_height() - 55  # Account for frame title and padding
                
                # Use reasonable minimums
                available_width = max(available_width, 400)
                available_height = max(available_height, 300)
            except:
                # Fallback if frame dimensions not available
                available_width = 800
                available_height = 600

            # Calculate scaling factor based on available space
            width_ratio = available_width / img_pil.width
            height_ratio = available_height / img_pil.height
            scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
            
            new_width = int(img_pil.width * scale_factor)
            new_height = int(img_pil.height * scale_factor)
            
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if ImageTk:
                img_tk = ImageTk.PhotoImage(img_pil)
                image_label.configure(image=img_tk)
                image_label.image = img_tk  # Keep a reference
            else:
                # Fallback: Save image and display path
                temp_path = "/tmp/temp_preview.jpg"
                img_pil.save(temp_path)
                image_label.configure(text=f"Image saved to: {temp_path}")

# Set ignored match parameters from GUI input
def set_parameters():
    global ignored_matches, ignored_matches_add
    try:
        c.IGNORED_MATCH_POSITIONS = int(ignored_matches.get())
        
        ignored_add_text = ignored_matches_add.get().strip()
        if ignored_add_text:
            # Filter out 0 and negative values from additional ignored positions
            c.ADDITIONAL_IGNORED_POSITIONS = [int(i.strip()) for i in ignored_add_text.split(",") if i.strip() and int(i.strip()) > 0]
        else:
            c.ADDITIONAL_IGNORED_POSITIONS = []
        
        if not run_on_mobile:
            update_image()
            messagebox.showinfo("Success", "Parameters updated successfully!")
    except ValueError as e:
        if not run_on_mobile:
            messagebox.showerror("Error", f"Invalid input: {e}")

# Save all settings to configuration file
def save_settings():
    global setting_entries
    try:
        # Get values from entry fields
        settings_values = {}
        for config_key, entry in setting_entries.items():
            value = entry.get()
            if config_key in ["SIMILARITY_THRESHOLD"]:
                settings_values[config_key] = float(value)
            else:
                settings_values[config_key] = int(value)
        
        # Create configuration dictionary
        variables_dict = {
            "RUN_ON_MOBILE": run_on_mobile,
            "IGNORED_MATCH_POSITIONS": c.IGNORED_MATCH_POSITIONS,
            "ADDITIONAL_IGNORED_POSITIONS": c.ADDITIONAL_IGNORED_POSITIONS,
            "ROI_TOP": c.ROI_TOP,
            "ROI_BOTTOM": c.ROI_BOTTOM,
            "ROI_PADDING": c.ROI_PADDING,
            "ENG_TOP": c.ENG_TOP,
            "ENG_BOTTOM": c.ENG_BOTTOM,
            "ENG_LEFT": c.ENG_LEFT,
            "ENG_RIGHT": c.ENG_RIGHT,
            "GO_TOP": c.GO_TOP,
            "GO_LEFT": c.GO_LEFT,
            "EX_TOP": c.EX_TOP,
            "EX_LEFT": c.EX_LEFT,
            "DEL_TOP": c.DEL_TOP,
            "DEL_BTN_TOP": c.DEL_BTN_TOP,
            "GRID_PADDING": c.GRID_PADDING,
            "DEL_BTN_SPACING": c.DEL_BTN_SPACING,
            "DEL_BTN_PADDING_RIGHT": c.DEL_BTN_PADDING_RIGHT,
            **settings_values
        }
        
        # Check if additional attributes exist
        for attr in ["R_GEN_TOP", "R_GEN_LEFT", "L_GEN_TOP", "L_GEN_LEFT"]:
            if hasattr(c, attr):
                variables_dict[attr] = getattr(c, attr)
        
        # Save to file
        json_data = json.dumps(variables_dict, indent=4)
        with open(c.config_path, "w") as json_file:
            json_file.write(json_data)
        
        if run_on_mobile:
            time.sleep(1)
            exit()
        else:
            messagebox.showinfo("Success", "Settings saved successfully!")
            
    except Exception as e:
        if not run_on_mobile:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

def create_cross_control(parent, title, up_cmd, down_cmd, left_cmd=None, right_cmd=None):
    """Create a cross-shaped control widget"""
    frame = ttk.LabelFrame(parent, text=title)
    frame.pack(fill=tk.X, pady=5, padx=5)
    
    # Create a grid for the cross layout
    control_frame = ttk.Frame(frame)
    control_frame.pack(pady=10)
    
    # Up button
    ttk.Button(control_frame, text="↑", command=up_cmd, width=3).grid(row=0, column=1, padx=2, pady=2)
    
    # Left and Right buttons (if provided)
    if left_cmd and right_cmd:
        ttk.Button(control_frame, text="←", command=left_cmd, width=3).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(control_frame, text="→", command=right_cmd, width=3).grid(row=1, column=2, padx=2, pady=2)
    
    # Down button
    ttk.Button(control_frame, text="↓", command=down_cmd, width=3).grid(row=2, column=1, padx=2, pady=2)
    
    return frame

def create_horizontal_control(parent, title, left_cmd, right_cmd):
    """Create a horizontal-only control widget (only left/right arrows)"""
    frame = ttk.LabelFrame(parent, text=title)
    frame.pack(fill=tk.X, pady=5, padx=5)
    
    # Create a horizontal layout
    control_frame = ttk.Frame(frame)
    control_frame.pack(pady=10)
    
    # Left and Right buttons only
    ttk.Button(control_frame, text="←", command=left_cmd, width=3).pack(side=tk.LEFT, padx=2)
    ttk.Button(control_frame, text="→", command=right_cmd, width=3).pack(side=tk.LEFT, padx=2)
    
    return frame

# Create the main tkinter GUI for desktop use
def create_tkinter_gui():
    global root, image_label, right_frame, ignored_matches, ignored_matches_add, setting_entries
    
    root = tk.Tk()
    root.title("Merge Boss Configurator")
    root.geometry("1200x800")
    
    # Main container with left and right panels
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left panel for controls and settings
    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
    
    # Right panel for image
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Event handler for window resize
    def on_window_resize(event):
        # Only update if the event is for the main window
        if event.widget == root:
            root.after(100, update_image)  # Delay to avoid too frequent updates
    
    root.bind('<Configure>', on_window_resize)
    
    # Left panel with scrollbar
    canvas = tk.Canvas(left_frame)
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Bind mousewheel to canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Title
    title = ttk.Label(scrollable_frame, text="Merge Boss Configurator", font=("Arial", 16, "bold"))
    title.pack(pady=10)
    
    subtitle = ttk.Label(scrollable_frame, 
                       text="Select a screenshot of the game to tune parameters",
                       wraplength=400)
    subtitle.pack(pady=5)
    
    # Screenshot section
    screenshot_frame = ttk.Frame(scrollable_frame)
    screenshot_frame.pack(fill=tk.X, pady=5)
    
    screenshot_btn = ttk.Button(screenshot_frame, text="Pick screenshot", command=pick_screenshot)
    screenshot_btn.pack(side=tk.LEFT, padx=5)
    
    load_screenshot_btn = ttk.Button(screenshot_frame, text="Load screenshot", command=load_screenshot)
    load_screenshot_btn.pack(side=tk.LEFT, padx=5)
    
    # Control panels with tabs
    controls_notebook = ttk.Notebook(scrollable_frame)
    controls_notebook.pack(fill=tk.X, pady=10)
    
    # Tab 1: ROI & Grid Controls
    roi_tab = ttk.Frame(controls_notebook)
    controls_notebook.add(roi_tab, text="ROI & Grid")
    
    roi_controls_frame = ttk.Frame(roi_tab)
    roi_controls_frame.pack(fill=tk.X, pady=10)
    
    # ROI Controls
    roi_frame = ttk.Frame(roi_controls_frame)
    roi_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    
    create_cross_control(roi_frame, "ROI Top", 
                        lambda: adjust_config('ROI_TOP', -0.001),
                        lambda: adjust_config('ROI_TOP', 0.001))
    
    create_cross_control(roi_frame, "ROI Bottom", 
                        lambda: adjust_config('ROI_BOTTOM', -0.001),
                        lambda: adjust_config('ROI_BOTTOM', 0.001))
    
    create_horizontal_control(roi_frame, "ROI Padding",
                        lambda: adjust_config('ROI_PADDING', -0.001),
                        lambda: adjust_config('ROI_PADDING', 0.001))

    create_cross_control(roi_frame, "Grid Padding",
                        lambda: adjust_config('GRID_PADDING', 0.001),
                        lambda: adjust_config('GRID_PADDING', -0.001))
    
    # Tab 2: Button Controls
    buttons_tab = ttk.Frame(controls_notebook)
    controls_notebook.add(buttons_tab, text="Buttons & Controls")
    
    buttons_controls_frame = ttk.Frame(buttons_tab)
    buttons_controls_frame.pack(fill=tk.X, pady=10)
    
    # Energy Controls
    energy_frame = ttk.Frame(buttons_controls_frame)
    energy_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    
    create_cross_control(energy_frame, "Energy Position",
                        lambda: adjust_energy_position(-0.002),
                        lambda: adjust_energy_position(0.002),
                        lambda: adjust_energy_horizontal(-0.002),
                        lambda: adjust_energy_horizontal(0.002))

    create_cross_control(energy_frame, "Left Generator",
                        lambda: adjust_config('L_GEN_TOP', -0.002),
                        lambda: adjust_config('L_GEN_TOP', 0.002),
                        lambda: adjust_config('L_GEN_LEFT', -0.002),
                        lambda: adjust_config('L_GEN_LEFT', 0.002))

    create_cross_control(energy_frame, "X Button",
                        lambda: adjust_config('EX_TOP', -0.002),
                        lambda: adjust_config('EX_TOP', 0.002),
                        lambda: adjust_config('EX_LEFT', -0.002),
                        lambda: adjust_config('EX_LEFT', 0.002))

    # Button Controls
    game_buttons_frame = ttk.Frame(buttons_controls_frame)
    game_buttons_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    
    create_cross_control(game_buttons_frame, "GO Button",
                        lambda: adjust_config('GO_TOP', -0.002),
                        lambda: adjust_config('GO_TOP', 0.002),
                        lambda: adjust_config('GO_LEFT', -0.002),
                        lambda: adjust_config('GO_LEFT', 0.002))

    create_cross_control(game_buttons_frame, "Right Generator",
                        lambda: adjust_config('R_GEN_TOP', -0.002),
                        lambda: adjust_config('R_GEN_TOP', 0.002),
                        lambda: adjust_config('R_GEN_LEFT', -0.002),
                        lambda: adjust_config('R_GEN_LEFT', 0.002))
    
    # Delivery Controls
    delivery_frame = ttk.Frame(buttons_controls_frame)
    delivery_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    
    create_cross_control(delivery_frame, "Delivery Swipe",
                        lambda: adjust_config('DEL_TOP', -0.002),
                        lambda: adjust_config('DEL_TOP', 0.002))
    
    create_cross_control(delivery_frame, "Delivery Button Position",
                        lambda: adjust_config('DEL_BTN_TOP', -0.002),
                        lambda: adjust_config('DEL_BTN_TOP', 0.002),
                        lambda: adjust_config('DEL_BTN_PADDING_RIGHT', 0.002),
                        lambda: adjust_config('DEL_BTN_PADDING_RIGHT', -0.002))

    create_horizontal_control(delivery_frame, "Delivery Button Spacing",
                        lambda: adjust_config('DEL_BTN_SPACING', 0.002),
                        lambda: adjust_config('DEL_BTN_SPACING', -0.002))

    # Image viewer in right panel
    image_frame = ttk.LabelFrame(right_frame, text="Screenshot Preview")
    image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Simple image label - no complex canvas needed for scaled images
    image_label = ttk.Label(image_frame, text="Load a screenshot to see preview", 
                           background='white', anchor='center')
    image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Parameters section
    param_frame = ttk.LabelFrame(scrollable_frame, text="Grid Parameters")
    param_frame.pack(fill=tk.X, pady=10, padx=10)
    
    # Ignored matches
    ttk.Label(param_frame, text="Ignore first N:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    ignored_matches = ttk.Entry(param_frame, width=42)
    ignored_matches.insert(0, str(c.IGNORED_MATCH_POSITIONS))
    ignored_matches.grid(row=0, column=1, padx=5, pady=2)
    
    # Additional ignored
    ttk.Label(param_frame, text="Ignore list:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    ignored_matches_add = ttk.Entry(param_frame, width=42)
    ignored_matches_add.insert(0, ",".join(map(str, c.ADDITIONAL_IGNORED_POSITIONS)))
    ignored_matches_add.grid(row=1, column=1, padx=5, pady=2)
    
    set_param_btn = ttk.Button(param_frame, text="Set Parameters", command=set_parameters)
    set_param_btn.grid(row=2, column=0, columnspan=2, pady=10)
    
    # Settings section
    settings_frame = ttk.LabelFrame(scrollable_frame, text="Game Settings")
    settings_frame.pack(fill=tk.X, pady=10, padx=10)
    
    # Create entry fields for settings
    settings = [
        ("Minimum energy level:", "MIN_ENERGY_LEVEL"),
        ("Maximum farm actions:", "MAX_FARM_SESSIONS"),
        ("Similarity threshold:", "SIMILARITY_THRESHOLD"),
        ("Generator min groups:", "MAX_GENERATOR_GROUP_NUMBERS"),
        ("Minimum blank spaces:", "MIN_SPACES_ON_BOARD")
    ]
    
    for i, (label, config_key) in enumerate(settings):
        ttk.Label(settings_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(settings_frame, width=10)
        entry.insert(0, str(getattr(c, config_key)))
        entry.grid(row=i, column=1, padx=5, pady=2)
        setting_entries[config_key] = entry
    
    # Save button
    save_btn = ttk.Button(scrollable_frame, text="Save settings", command=save_settings)
    save_btn.pack(pady=10)

def main():
    if run_on_mobile:
        # Termux:GUI implementation for mobile devices
        with tg.Connection() as connection:
            activity = tg.Activity(connection)
            rootLinear = tg.LinearLayout(activity)
            title = tg.TextView(activity, "Merge Boss Configurator", rootLinear)
            title.settextsize(24)
            title.setmargin(5)
            title.setlinearlayoutparams(0)
            title.setheight(tg.View.WRAP_CONTENT)

            scrollView = tg.NestedScrollView(activity, rootLinear)
            scrollLinear = tg.LinearLayout(activity, scrollView)

            subtitle = tg.TextView(
                activity,
                "Select an screenshot of the game to tune parameters like ROI, padding, and button positions for your game.",
                scrollLinear,
            )
            subtitle.settextsize(16)
            subtitle.setmargin(5)
            subtitle.setlinearlayoutparams(0)
            subtitle.setheight(tg.View.WRAP_CONTENT)

            screenshotLinearHorizontal = tg.LinearLayout(activity, scrollLinear, False)

            screenshot_btn = create_button(
                activity, "Pick screenshot", screenshotLinearHorizontal
            )
            load_screenshot_btn = create_button(
                activity, "Load screenshot", screenshotLinearHorizontal
            )

            positionHorizontalScroll = tg.HorizontalScrollView(activity, scrollLinear)
            positionLinearHorizontal = tg.LinearLayout(
                activity, positionHorizontalScroll, False
            )

            _ = create_button(activity, "ROI", positionLinearHorizontal)
            top_pl_roi_btn = create_button(activity, "↑+", positionLinearHorizontal)
            top_mn_roi_btn = create_button(activity, "↑-", positionLinearHorizontal)
            bot_pl_roi_btn = create_button(activity, "↓+", positionLinearHorizontal)
            bot_mn_roi_btn = create_button(activity, "↓-", positionLinearHorizontal)
            left_pl_roi_btn = create_button(activity, "↔+", positionLinearHorizontal)
            left_mn_roi_btn = create_button(activity, "↔-", positionLinearHorizontal)
            grid_pl_roi_btn = create_button(activity, "G+", positionLinearHorizontal)
            grid_mn_roi_btn = create_button(activity, "G-", positionLinearHorizontal)

            _ = create_button(activity, "ENERGY", positionLinearHorizontal)
            top_mn_eng_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_pl_eng_btn = create_button(activity, "↓", positionLinearHorizontal)
            left_mn_eng_btn = create_button(activity, "←", positionLinearHorizontal)
            left_pl_eng_btn = create_button(activity, "→", positionLinearHorizontal)

            _ = create_button(activity, "GO", positionLinearHorizontal)
            top_pl_go_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_go_btn = create_button(activity, "↓", positionLinearHorizontal)
            left_pl_go_btn = create_button(activity, "←", positionLinearHorizontal)
            left_mn_go_btn = create_button(activity, "→", positionLinearHorizontal)

            _ = create_button(activity, "X", positionLinearHorizontal)
            top_pl_ex_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_ex_btn = create_button(activity, "↓", positionLinearHorizontal)
            left_pl_ex_btn = create_button(activity, "←", positionLinearHorizontal)
            left_mn_ex_btn = create_button(activity, "→", positionLinearHorizontal)
            
            _ = create_button(activity, "D. SWIPE", positionLinearHorizontal)
            top_pl_delswp_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_delswp_btn = create_button(activity, "↓", positionLinearHorizontal)
            
            _ = create_button(activity, "D. BTN", positionLinearHorizontal)
            top_pl_delb_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_delb_btn = create_button(activity, "↓", positionLinearHorizontal)
            spce_pl_delb_btn = create_button(activity, "↹", positionLinearHorizontal)
            spce_mn_delb_btn = create_button(activity, "⇎", positionLinearHorizontal)
            padr_pl_delb_btn = create_button(activity, "←", positionLinearHorizontal)
            padr_mn_delb_btn = create_button(activity, "→", positionLinearHorizontal)

            _ = create_button(activity, "R. Gen. BTN ", positionLinearHorizontal)
            top_pl_r_gen_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_r_gen_btn = create_button(activity, "↓", positionLinearHorizontal)
            left_pl_r_gen_btn = create_button(activity, "←", positionLinearHorizontal)
            left_mn_r_gen_btn = create_button(activity, "→", positionLinearHorizontal)

            _ = create_button(activity, "L. Gen. BTN ", positionLinearHorizontal)
            top_pl_l_gen_btn = create_button(activity, "↑", positionLinearHorizontal)
            top_mn_l_gen_btn = create_button(activity, "↓", positionLinearHorizontal)
            left_pl_l_gen_btn = create_button(activity, "←", positionLinearHorizontal)
            left_mn_l_gen_btn = create_button(activity, "→", positionLinearHorizontal)

            _, rootHeight = rootLinear.getdimensions()
            image_viewer = tg.ImageView(activity, scrollLinear)
            image_viewer.setlinearlayoutparams(0)
            image_viewer.setheight(rootHeight - int(rootHeight / 7), True)
            
            paramGridLayout = tg.GridLayout(activity, 2, 3, scrollLinear)
            
            ignore_matches_txt = tg.TextView(activity, "Ignore first N", paramGridLayout)
            ignore_matches_txt.setgridlayoutparams(0, 0)
            ignore_matches_txt.setwidth(100)
            
            ignore_matches_add_txt = tg.TextView(activity, "Ignore list", paramGridLayout)
            ignore_matches_add_txt.setgridlayoutparams(0, 1)
            ignore_matches_add_txt.setwidth(100)

            ignored_matches = tg.EditText(
                activity,
                str(c.IGNORED_MATCH_POSITIONS),
                paramGridLayout,
                singleline=True,
                inputtype="number",
            )
            ignored_matches.setgridlayoutparams(1, 0)
            ignored_matches.setwidth(100)
            
            ignored_matches_add = tg.EditText(
                activity,
                ",".join(map(str, c.ADDITIONAL_IGNORED_POSITIONS)),
                paramGridLayout,
                singleline=True,
                inputtype="number",
            )
            ignored_matches_add.setgridlayoutparams(1, 1)
            ignored_matches_add.setwidth(100)

            set_param_btn = create_button(activity, "Set", paramGridLayout)
            set_param_btn.setgridlayoutparams(0, 2, 2, 1)

            settingsGridLayout = tg.GridLayout(activity, 6, 2, scrollLinear)
            min_eng_lvl_txt = tg.TextView(activity, "Minimum energy level", settingsGridLayout)
            min_eng_lvl_txt.setgridlayoutparams(0, 0)
            min_eng_lvl_txt.setwidth(145)
            max_farm_act_txt = tg.TextView(
                activity, "Maximum farm actions", settingsGridLayout
            )
            max_farm_act_txt.setgridlayoutparams(0, 1)
            max_farm_act_txt.setwidth(145)

            min_eng_lvl = tg.EditText(
                activity,
                str(c.MIN_ENERGY_LEVEL),
                settingsGridLayout,
                singleline=True,
                inputtype="number",
            )
            min_eng_lvl.setgridlayoutparams(1, 0)
            min_eng_lvl.setwidth(145)
            max_farm_act = tg.EditText(
                activity,
                str(c.MAX_FARM_SESSIONS),
                settingsGridLayout,
                singleline=True,
                inputtype="number",
            )
            max_farm_act.setgridlayoutparams(1, 1)
            max_farm_act.setwidth(145)

            sim_thresh_txt = tg.TextView(activity, "Similarity threshold", settingsGridLayout)
            sim_thresh_txt.setgridlayoutparams(2, 0)
            sim_thresh_txt.setwidth(145)
            gen_min_groups_txt = tg.TextView(
                activity, "Press generator when minimum groups number", settingsGridLayout
            )
            gen_min_groups_txt.setgridlayoutparams(2, 1)
            gen_min_groups_txt.setwidth(145)

            sim_thresh = tg.EditText(
                activity,
                str(c.SIMILARITY_THRESHOLD),
                settingsGridLayout,
                singleline=True,
                inputtype="number",
            )
            sim_thresh.setgridlayoutparams(3, 0)
            sim_thresh.setwidth(145)
            gen_min_groups = tg.EditText(
                activity,
                str(c.MAX_GENERATOR_GROUP_NUMBERS),
                settingsGridLayout,
                singleline=True,
                inputtype="number",
            )
            gen_min_groups.setgridlayoutparams(3, 1)
            gen_min_groups.setwidth(145)
            
            min_blank_txt = tg.TextView(activity, "Minimum blank spaces", settingsGridLayout)
            min_blank_txt.setgridlayoutparams(4, 0)
            min_blank_txt.setwidth(145)
            
            min_blank_spc = tg.EditText(
                activity,
                str(c.MIN_SPACES_ON_BOARD),
                settingsGridLayout,
                singleline=True,
                inputtype="number",
            )
            min_blank_spc.setgridlayoutparams(5, 0)
            min_blank_spc.setwidth(145)

            save_btn = create_button(activity, "Save settings", scrollLinear)

            def handle_config_update(btn_id, attr, delta):
                """Helper to update config and refresh image"""
                if isinstance(attr, list):
                    for a in attr:
                        setattr(c, a, getattr(c, a) + delta)
                else:
                    setattr(c, attr, getattr(c, attr) + delta)
                annotate_and_load_img(screenshot_path, image_viewer)

            screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
            for event in connection.events():
                if event.type == tg.Event.click and event.value["id"] == screenshot_btn:
                    os.system(f"termux-storage-get {screenshot_path}")
                elif event.type == tg.Event.click and event.value["id"] == load_screenshot_btn:
                    annotate_and_load_img(screenshot_path, image_viewer)
                elif event.type == tg.Event.click:
                    btn_id = event.value["id"]
                    # ROI controls
                    if btn_id == top_pl_roi_btn:
                        handle_config_update(btn_id, "ROI_TOP", 0.001)
                    elif btn_id == top_mn_roi_btn:
                        handle_config_update(btn_id, "ROI_TOP", -0.001)
                    elif btn_id == bot_pl_roi_btn:
                        handle_config_update(btn_id, "ROI_BOTTOM", 0.001)
                    elif btn_id == bot_mn_roi_btn:
                        handle_config_update(btn_id, "ROI_BOTTOM", -0.001)
                    elif btn_id == left_pl_roi_btn:
                        handle_config_update(btn_id, "ROI_PADDING", 0.001)
                    elif btn_id == left_mn_roi_btn:
                        handle_config_update(btn_id, "ROI_PADDING", -0.001)
                    elif btn_id == grid_pl_roi_btn:
                        handle_config_update(btn_id, "GRID_PADDING", 0.001)
                    elif btn_id == grid_mn_roi_btn:
                        handle_config_update(btn_id, "GRID_PADDING", -0.001)
                    # Energy controls  
                    elif btn_id == top_pl_eng_btn:
                        handle_config_update(btn_id, ["ENG_TOP", "ENG_BOTTOM"], 0.002)
                    elif btn_id == top_mn_eng_btn:
                        handle_config_update(btn_id, ["ENG_TOP", "ENG_BOTTOM"], -0.002)
                    elif btn_id == left_pl_eng_btn:
                        handle_config_update(btn_id, ["ENG_LEFT", "ENG_RIGHT"], 0.002)
                    elif btn_id == left_mn_eng_btn:
                        handle_config_update(btn_id, ["ENG_LEFT", "ENG_RIGHT"], -0.002)
                    # GO controls
                    elif btn_id == top_mn_go_btn:
                        handle_config_update(btn_id, "GO_TOP", 0.002)
                    elif btn_id == top_pl_go_btn:
                        handle_config_update(btn_id, "GO_TOP", -0.002)
                    elif btn_id == left_mn_go_btn:
                        handle_config_update(btn_id, "GO_LEFT", 0.002)
                    elif btn_id == left_pl_go_btn:
                        handle_config_update(btn_id, "GO_LEFT", -0.002)
                    # EX controls
                    elif btn_id == top_mn_ex_btn:
                        handle_config_update(btn_id, "EX_TOP", 0.002)
                    elif btn_id == top_pl_ex_btn:
                        handle_config_update(btn_id, "EX_TOP", -0.002)
                    elif btn_id == left_mn_ex_btn:
                        handle_config_update(btn_id, "EX_LEFT", 0.002)
                    elif btn_id == left_pl_ex_btn:
                        handle_config_update(btn_id, "EX_LEFT", -0.002)
                    # DEL controls
                    elif btn_id == top_pl_delswp_btn:
                        handle_config_update(btn_id, "DEL_TOP", -0.002)
                    elif btn_id == top_mn_delswp_btn:
                        handle_config_update(btn_id, "DEL_TOP", 0.002)
                    elif btn_id == top_pl_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_TOP", -0.002)
                    elif btn_id == top_mn_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_TOP", 0.002)
                    elif btn_id == spce_pl_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_SPACING", 0.002)
                    elif btn_id == spce_mn_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_SPACING", -0.002)
                    elif btn_id == padr_pl_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_PADDING_RIGHT", -0.002)
                    elif btn_id == padr_mn_delb_btn:
                        handle_config_update(btn_id, "DEL_BTN_PADDING_RIGHT", 0.002)
                    # Generator controls
                    elif btn_id == top_pl_r_gen_btn:
                        handle_config_update(btn_id, "R_GEN_TOP", -0.002)
                    elif btn_id == top_mn_r_gen_btn:
                        handle_config_update(btn_id, "R_GEN_TOP", 0.002)
                    elif btn_id == left_pl_r_gen_btn:
                        handle_config_update(btn_id, "R_GEN_LEFT", -0.002)
                    elif btn_id == left_mn_r_gen_btn:
                        handle_config_update(btn_id, "R_GEN_LEFT", 0.002)
                    elif btn_id == top_pl_l_gen_btn:
                        handle_config_update(btn_id, "L_GEN_TOP", -0.002)
                    elif btn_id == top_mn_l_gen_btn:
                        handle_config_update(btn_id, "L_GEN_TOP", 0.002)
                    elif btn_id == left_pl_l_gen_btn:
                        handle_config_update(btn_id, "L_GEN_LEFT", -0.002)
                    elif btn_id == left_mn_l_gen_btn:
                        handle_config_update(btn_id, "L_GEN_LEFT", 0.002)
                    # Special buttons
                    elif btn_id == set_param_btn:
                        try:
                            c.IGNORED_MATCH_POSITIONS = int(ignored_matches.gettext())
                            
                            ignored_add_text = ignored_matches_add.gettext().strip()
                            if ignored_add_text:
                                # Filter out 0 and negative values from additional ignored positions
                                c.ADDITIONAL_IGNORED_POSITIONS = [
                                    int(i.strip()) for i in ignored_add_text.split(",") if i.strip() and int(i.strip()) > 0
                                ]
                            else:
                                c.ADDITIONAL_IGNORED_POSITIONS = []
                            
                            annotate_and_load_img(screenshot_path, image_viewer)
                        except ValueError:
                            pass
                    elif event.value["id"] == save_btn:
                        try:
                            # Create configuration dictionary
                            variables_dict = {
                                "RUN_ON_MOBILE": run_on_mobile,
                                "IGNORED_MATCH_POSITIONS": c.IGNORED_MATCH_POSITIONS,
                                "ADDITIONAL_IGNORED_POSITIONS": c.ADDITIONAL_IGNORED_POSITIONS,
                                "ROI_TOP": c.ROI_TOP,
                                "ROI_BOTTOM": c.ROI_BOTTOM,
                                "ROI_PADDING": c.ROI_PADDING,
                                "ENG_TOP": c.ENG_TOP,
                                "ENG_BOTTOM": c.ENG_BOTTOM,
                                "ENG_LEFT": c.ENG_LEFT,
                                "ENG_RIGHT": c.ENG_RIGHT,
                                "GO_TOP": c.GO_TOP,
                                "GO_LEFT": c.GO_LEFT,
                                "EX_TOP": c.EX_TOP,
                                "EX_LEFT": c.EX_LEFT,
                                "DEL_TOP": c.DEL_TOP,
                                "DEL_BTN_TOP": c.DEL_BTN_TOP,
                                "GRID_PADDING": c.GRID_PADDING,
                                "DEL_BTN_SPACING": c.DEL_BTN_SPACING,
                                "DEL_BTN_PADDING_RIGHT": c.DEL_BTN_PADDING_RIGHT,
                                "MIN_ENERGY_LEVEL": int(min_eng_lvl.gettext()),
                                "MAX_FARM_SESSIONS": int(max_farm_act.gettext()),
                                "SIMILARITY_THRESHOLD": float(sim_thresh.gettext()),
                                "MAX_GENERATOR_GROUP_NUMBERS": int(gen_min_groups.gettext()),
                                "MIN_SPACES_ON_BOARD": int(min_blank_spc.gettext()),
                                "R_GEN_TOP": c.R_GEN_TOP,
                                "R_GEN_LEFT": c.R_GEN_LEFT,
                                "L_GEN_TOP": c.L_GEN_TOP,
                                "L_GEN_LEFT": c.L_GEN_LEFT,
                            }

                            # Save to file
                            json_data = json.dumps(variables_dict, indent=4)
                            with open(c.config_path, "w") as json_file:
                                json_file.write(json_data)
                            
                            time.sleep(1)
                            exit()
                        except Exception:
                            pass
    else:
        create_tkinter_gui()
        root.mainloop()

if __name__ == "__main__":
    main()