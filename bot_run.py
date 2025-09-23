from ppadb.client import Client as AdbClient
import cv2
import numpy as np
import time
import configuration as c
import threading
import os


# Import Termux:GUI to display overlay if script is running on Android
if c.RUN_ON_MOBILE:
    import termuxgui as tg
    

# A function that generates a button with specified text, layout, and optional width.
def create_overlay_button(activity, text, layout, width=40):
    button = tg.Button(activity, text, layout)
    button.settextsize(12)
    button.setlinearlayoutparams(1)
    if width:
        button.setwidth(width)
        button.setheight(width)

    return button


# Create an overlay with buttons to control bot state
def display_overlay_on_android(height, connection):
    activity = tg.Activity(connection, tid=110, overlay=True)
    activity.setposition(9999, (c.DEL_TOP / 4.5) * height)
    activity.keepscreenon(True)
    activity.sendoverlayevents(False)
    
    rootLinear = tg.LinearLayout(activity, vertical=False)
    
    play_pause_btn = create_overlay_button(activity, "⏯️", rootLinear) 
    farm_btn = create_overlay_button(activity, "🧑‍🌾", rootLinear) 
    exit_btn = create_overlay_button(activity, "❌", rootLinear) 
    
    time.sleep(1)
            
    return play_pause_btn, farm_btn, exit_btn
    
    
# Set flags for next action when button press
pause_flag, farm_flag, exit_flag = [False] * 3
def action_on_overlay_button_press(connection, play_pause_id, farm_id, exit_id):
    global pause_flag, farm_flag, exit_flag
    for event in connection.events():
        if event and event.type == tg.Event.click and event.value["id"] == play_pause_id:
            pause_flag = not pause_flag
            if pause_flag:
                connection.toast("Bot is pausing")
            else:
                connection.toast("Bot starting")
        if event and event.type == tg.Event.click and event.value["id"] == farm_id:
            farm_flag = True and pause_flag
            if farm_flag:
                connection.toast("Starting energy farm")
            else:
                connection.toast("Please pause before farm")
        if event and event.type == tg.Event.click and event.value["id"] == exit_id:
            exit_flag = True
            connection.toast("Closing bot")


# Check the state of the button flags
def do_button_flags(img, device):
    global pause_flag, farm_flag, exit_flag 
    while pause_flag:
        time.sleep(0.5)
        # If bot is paused and manual farm energy is requested, start to farm
        if farm_flag:
            print("Starting to farm energy.")
            # Touch the game window to focus back on it
            device.input_tap(img.shape[1] // 2, img.shape[0] // 2)
            farm_energy(img, device)
            farm_flag = False
        if exit_flag:
            exit()
    if exit_flag:
        exit()            


# Get and image from ADB and transform it to opencv image
def get_screen_capture(device):
    result = device.screencap()
    img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    return img


# Applies Sobel edge detection to highlight edges in the image, with a user-defined threshold.
def sobel_edge_detector(img, threshold=50):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    _, binary_edge = cv2.threshold(grad_norm, threshold, 255, cv2.THRESH_BINARY)

    return binary_edge


# Display the extracted images after applying apply_processing
DISPLAY_EXTRACTED_IMGS = True and not c.RUN_ON_MOBILE

# Display the annotated image
DISPLAY_ANNOTATED_IMGS = True and not c.RUN_ON_MOBILE

if c.CHECK_ENERGY_LEVEL:
    import pytesseract

    print("Make sure you have Tesseract installed on the system and added to PATH")
    

# Applies Sobel edge detection to highlight edges in the image.
def apply_processing(img, sob_thresh=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract edges using Sobel
    sobel_edges = sobel_edge_detector(gray, sob_thresh)
    return sobel_edges


# Creates a grid of rectangular contours within the specified region of interest (ROI) with padding.
def generate_grid_contours(img, roi, padding=5):
    height, width, _ = img.shape
    roi_min, roi_max, width_padding = roi
    max_rows, max_col = 5, 7
    square_size = (roi_max - roi_min) // max_rows
    contours = []

    # Convert relative padding to absolute pixels
    padding_pixels = int(padding * min(width, height))

    # Draw rectangles mesh and find contours with padding
    for row in range(max_rows):
        for col in range(max_col):
            x = (col * square_size) + int(width_padding) + padding_pixels
            y = roi_min + row * square_size + padding_pixels
            contour = np.array(
                [
                    (x, y),
                    (x + square_size - 2 * padding_pixels, y),
                    (x + square_size - 2 * padding_pixels, y + square_size - 2 * padding_pixels),
                    (x, y + square_size - 2 * padding_pixels),
                ]
            )
            contours.append(contour)

    return contours


# Extracts images from a list of contours. Applies image processing (Sobel edge detection) and morphological dilation for better blob extraction.
img_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def extract_imgs_from_contours(img, contours):
    imgs_list = []
    proc_img = apply_processing(img)
    # Apply morphological dilation to the processed image
    proc_img = cv2.dilate(proc_img, img_dilation_kernel, iterations=1)
    count_blanks = 0
    for contour in contours:
        # Extract blobs from each contour using the adaptive threshold function
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = proc_img[y : y + h, x : x + w]
        # Check if the image is blank and count blank images to see if there's space left on grid
        is_blank = is_blank_img(cropped_img)
        if is_blank:
            count_blanks += 1
        imgs_list.append([cropped_img, is_blank])

    return imgs_list, count_blanks


# Determines if an image is blank based on the number of non-zero pixels, using a specified threshold.
def is_blank_img(img, threshold_pixels=200):
    # Count the number of non-zero pixels
    non_zero_count = np.sum(img == 255)

    # Check if the count is below the threshold
    return non_zero_count < threshold_pixels


# Compares two images by finding differing pixels and calculates a normalized similarity metric.
def compare_imgs(img1, img2):
    height, width = img1.shape

    # Find pixels that differ between the two images
    diff_img = np.bitwise_xor(img1, img2)

    # Count the number of differing pixels
    diff_pixels_cnt = np.count_nonzero(diff_img)

    normalized_similarity = 1 - (diff_pixels_cnt / (height * width))

    return normalized_similarity


# Groups similar images based on the specified similarity threshold, ignoring blank and ignored positions. Returns a list of grouped items.
def group_similar_imgs(imgs, compare_threshold=0.8):
    grouped_items = []
    visited = set()

    for i, img1 in enumerate(imgs):
        # Check if image is blank, or if it's in the ignored positions
        is_ignored_sequential = c.IGNORED_MATCH_POSITIONS > 0 and i < c.IGNORED_MATCH_POSITIONS
        is_ignored_additional = (i + 1) in c.ADDITIONAL_IGNORED_POSITIONS
        
        if img1[1] or is_ignored_sequential or is_ignored_additional:
            visited.add(i)
            continue

        if i not in visited:
            group = [i]
            found_match = False  # Flag to check if any similar blob is found

            for j, img2 in enumerate(imgs):
                # Check if image is blank, or if it's in the ignored positions
                is_ignored_sequential_j = c.IGNORED_MATCH_POSITIONS > 0 and j < c.IGNORED_MATCH_POSITIONS
                is_ignored_additional_j = (j + 1) in c.ADDITIONAL_IGNORED_POSITIONS
                
                if img2[1] or is_ignored_sequential_j or is_ignored_additional_j:
                    visited.add(j)
                    continue
                if i != j and j not in visited:
                    similarity = compare_imgs(img1[0], img2[0])
                    if similarity > compare_threshold:
                        group.append(j)
                        visited.add(j)
                        found_match = True

            # Add the group only if a match was found
            if found_match:
                grouped_items.append(group)
                for index in group:
                    visited.add(index)

    return grouped_items


# Annotates an image with marked regions of interest (ROI), ignored contours, and marked contours within groups.
# Contours are drawn with different colors and labeled with their respective group IDs.
def annotate_image(img, contours, groups, roi):
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

    # Unpack region of interest
    roi_min, roi_max, width_padding = roi

    # Mark ROI on image
    cv2.rectangle(
        img,
        (width_padding, roi_min),
        (width - width_padding, roi_max),
        (0, 255, 255),
        6,
    )

    # Mark energy position, X and "GO" button
    cv2.rectangle(img, (eng_left, eng_top), (eng_right, eng_bot), (255, 0, 255), 6)
    img = cv2.circle(img, (go_btn_left, go_btn_top), 20, (0, 255, 0), -1)
    img = cv2.circle(img, (close_btn_left, close_btn_top), 20, (0, 0, 255), -1)
    img = cv2.circle(img, (r_gen_btn_left, r_gen_btn_top), 25, (0, 255, 255), -1)
    img = cv2.circle(img, (l_gen_btn_left, l_gen_btn_top), 25, (255, 255, 0), -1)

    # Draw line and points for automatic delivery - Convert to relative coordinates
    img = cv2.line(
        img, (width // 2, delivery_top), (width, delivery_top), (255, 255, 0), 10
    )
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width * 2), delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width), delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (width - int(c.DEL_BTN_PADDING_RIGHT * width), delivery_btn_top), 20, (255, 50, 255), -1)

    # Draw ignored contours (only if IGNORED_MATCH_POSITIONS > 0)
    if c.IGNORED_MATCH_POSITIONS > 0:
        for ig in range(c.IGNORED_MATCH_POSITIONS):
            cv2.drawContours(img, [contours[ig]], 0, (0, 0, 255), 4)
        
    # Draw addititional ignored contours
    for ig in c.ADDITIONAL_IGNORED_POSITIONS:
        if ig > 0 and ig <= len(contours):  # Ensure valid position
            cv2.drawContours(img, [contours[ig - 1]], 0, (0, 0, 255), 4)

    # Mark contours and groups on image
    for group_id, contour_indices in enumerate(groups):
        color = (group_id * 30) % 255
        for index in contour_indices:
            contour = contours[index]
            cv2.drawContours(img, [contour], 0, (color, 127, 50), 3)
            cv2.putText(
                img,
                str(group_id),
                tuple(contour[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (color, 127, 50),
                4,
            )

    return img


# Swipes through elements within groups on a device, avoiding repeated swiping of positions.
def swipe_elements(device, contours, groups, roi):
    roi_min, roi_max, width_padding = roi
    already_swiped_positions = set()

    for group_id, contour_indices in enumerate(groups):
        for i in range(len(contour_indices) - 1):
            index1 = contour_indices[i]
            index2 = contour_indices[i + 1]

            contour1 = contours[index1]
            contour2 = contours[index2]

            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)

            # Check if the positions have already been swiped
            position1 = (x1 + w1 // 2, y1 + h1 // 2)
            position2 = (x2 + w2 // 2, y2 + h2 // 2)

            if (
                position1 in already_swiped_positions
                or position2 in already_swiped_positions
            ):
                continue  # Skip if either position has been swiped

            # Swipe from the center of the first contour to the center of the second contour
            device.input_swipe(
                x1 + w1 // 2, y1 + h1 // 2, x2 + w2 // 2, y2 + h2 // 2, 80
            )

            # Update the set of already swiped positions
            already_swiped_positions.update([position1, position2])

    if len(groups) > 0:
        # Touch the first element of the list after merge, because of contouring
        x1, y1, _, _ = cv2.boundingRect(contours[0])
        device.input_tap(x1, y1)


# Try to deliver generated items
def try_to_delivery(device, img_shape):
    height, width, _ = img_shape
    delivery_top = int(c.DEL_TOP * height)
    delivery_btn_top = int(c.DEL_BTN_TOP * height)

    # Swipe through the delivery list and try to press the "Delivery" button
    for i in range(3):
        device.input_tap(width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width * 2), delivery_btn_top)
        device.input_tap(width - int(c.DEL_BTN_PADDING_RIGHT * width) - int(c.DEL_BTN_SPACING * width), delivery_btn_top)
        device.input_tap(width - int(c.DEL_BTN_PADDING_RIGHT * width), delivery_btn_top)
        device.input_swipe(width - 100, delivery_top, width // 2, delivery_top, 40)

    # Go back
    for i in range(6):
        device.input_swipe(width // 2, delivery_top, width - 100, delivery_top, 40)


# Generates objects by clicking on specified generator positions.
def generate_objects(device, width, height, img, count_blanks):
    energy_left = get_energy_level(img)
    if energy_left <= c.MIN_ENERGY_LEVEL:
        print("No energy left")
        return False
    for i in range(energy_left, c.MIN_ENERGY_LEVEL, -1):
        print(f"Blank Fields: {count_blanks}, Energy left: {i}")
        if count_blanks < 2:
            return True
        if i % 2 == 0:
            device.input_tap(width * c.R_GEN_LEFT, height * c.R_GEN_TOP)
        else:
            device.input_tap(width * c.L_GEN_LEFT, height * c.L_GEN_TOP)
        count_blanks -= 1
        time.sleep(0.1)
    return True


# Resizes input image based on the specified max height.
def resize_image(image, max_height=720):
    # Get the original dimensions of the image
    if len(image.shape) == 3:
        original_height, original_width, _ = image.shape
    else:
        original_height, original_width = image.shape

    # Calculate the scaling factor to maintain aspect ratio
    scale_factor = max_height / original_height

    # Calculate the new dimensions
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# Extracts and returns the energy level from a given image. Utilizes Tesseract OCR for text extraction.
def get_energy_level(img):
    # Note: you should have Tesseract installed and set in path to use this function
    height, width, _ = img.shape
    x, y, x1, y1 = (
        int(width * c.ENG_LEFT),
        int(height * c.ENG_TOP),
        int(width * c.ENG_RIGHT),
        int(height * c.ENG_BOTTOM),
    )
    cropped = img[y:y1, x:x1]
    if c.IMAGE_DEBUG:
        cv2.imwrite("cropped.png", cropped)

    min_pixel = 120
    max_pixel = 300
    gap = 2
    padding = 3

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if c.IMAGE_DEBUG:
        _, old_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("old_thresh.png", old_thresh)
        cv2.floodFill(old_thresh, None, (5, 5), 0, flags=8)
        cv2.imwrite("old_finish.png", old_thresh)

    # Preprocessing & dilate
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if c.IMAGE_DEBUG:
        cv2.imwrite("new_thresh.png", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Invert for component analysis
    binary_inv = cv2.bitwise_not(dilate)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv)

    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_pixel <= area <= max_pixel:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Add padding
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            x2_pad = min(dilate.shape[1], x + w + padding)
            y2_pad = min(dilate.shape[0], y + h + padding)

            region = dilate[y_pad:y2_pad, x_pad:x2_pad]
            mask = (labels[y_pad:y2_pad, x_pad:x2_pad] == i).astype(np.uint8)

            # Dilate mask
            kernel_pad = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel_pad)

            # Store region + mask
            regions.append((x, region, mask_dilated))

    # Sort by X coordinate (from left to right)
    regions.sort(key=lambda r: r[0])

    # Calculate target dimensions
    heights = [r[1].shape[0] for r in regions]
    widths = [r[1].shape[1] for r in regions]
    total_width = sum(widths) + gap * (len(regions) - 1)
    max_height = max(heights)

    # Create new white image (grayscale)
    output = np.full((max_height, total_width), 255, dtype=np.uint8)

    # Insert regions
    current_x = 0
    for _, region, mask in regions:
        h, w = region.shape
        roi = output[0:h, current_x:current_x + w]
        roi[mask == 1] = region[mask == 1]
        current_x += w + gap

    if c.IMAGE_DEBUG:
        cv2.imwrite("new_finish.png", output)

    # Use pytesseract to extract numeric text from the preprocessed image
    custom_config = r"--oem 3 --psm 7 outputbase digits"  # Tesseract OCR configuration for numeric digits
    text = pytesseract.image_to_string(output, config=custom_config)

    if DISPLAY_EXTRACTED_IMGS:
        cv2.imshow("Extracted energy", output)
        print("Energy: ", text.strip())
        cv2.waitKey(5)

    try:
        return int(text.strip())
    except ValueError:
        print("Could not detect energy level, assuming 100, text: ", text.strip())
        return 100


# Automatically farm energy from tasks
def farm_energy(img, device):
    height, width, _ = img.shape
    # Open tasks menu
    device.input_tap(width * c.ENG_LEFT, height * c.ENG_TOP)
    time.sleep(3)
    for i in range(c.MAX_FARM_SESSIONS):
        # Hit the "Go" button and wait for X seconds
        device.input_tap(width * c.GO_LEFT, height * c.GO_TOP)
        # Wait for product list while check if bot still running
        elapsed_time = 0
        while elapsed_time < 17:
            check_if_should_exit(device)
            time.sleep(1)
            elapsed_time += 1
        device.input_keyevent("BACK")
        time.sleep(2)
        # Hit the "Claim" button
        device.input_tap(width * c.GO_LEFT, height * c.GO_TOP)
        time.sleep(3)
    # Exit tasks menu
    device.input_tap(width * c.EX_LEFT, height * c.EX_TOP)
    time.sleep(1)


# Combines a list of binary images into a grid with the specified number of columns and rows, just for debugging display.
def combine_binary_images(extracted_imgs, columns=7, rows=5):
    # Ensure that the number of images is consistent with the specified grid size
    if len(extracted_imgs) != (columns * rows):
        raise ValueError(
            f"Number of images ({len(extracted_imgs)}) is not compatible with the grid size ({columns}x{rows})."
        )

    # Resize images to have the same height (assuming they have the same width)
    height = extracted_imgs[0][0].shape[0]
    resized_imgs = [cv2.resize(img[0], (height, height)) for img in extracted_imgs]

    # Combine images into a grid
    combined_img = np.vstack(
        [
            np.hstack(resized_imgs[i : i + columns])
            for i in range(0, len(resized_imgs), columns)
        ]
    )

    return combined_img


# Checks if the current app running on the device is Aliexpress and the screen is on.
def check_app_in_foreground(device, target):
    result = device.shell("dumpsys window | grep -E 'mCurrentFocus'")
    if target in result or 'com.termux.gui' in result:
        return True

    return False


# Check if the space on the grid is sufficient
def check_if_space_left(count_blanks, grouped_items):
    if count_blanks + len(grouped_items) < c.MIN_SPACES_ON_BOARD:
        print("There's no space left on the grid and no swipe possible. Bot will exit.")
        exit()

# Check if the bot should terminate and exit
def check_if_should_exit(device):
    global exit_flag
    # Stop when user close Aliexpress app 
    if not check_app_in_foreground(device, c.TARGET_APP_PKG):
        print("Aliexpress app is not running anymore")
        exit()
    # Stop farming when user close the bot with button
    if exit_flag:
        exit()


# Waits for the Aliexpress app to be opened on the device.
def wait_for_ali_app(device):
    if check_app_in_foreground(device, c.TARGET_APP_PKG):
        print("Aliexpress app is running")
        return True
    else:
        print("Please open Aliexpress Merge Boss game. Waiting 15 seconds.")
        time.sleep(15)
        if not check_app_in_foreground(device, c.TARGET_APP_PKG):
            print("Merge Boss is not in focus. Exiting script")
            exit()
            
            
# These are for debugging and calibration purposes
def debug_display_img(img, grid_contours, grouped_items, roi, extracted_imgs):
    if DISPLAY_EXTRACTED_IMGS:
        display_extracted_img = combine_binary_images(extracted_imgs)
        res_display_extracted_img = resize_image(display_extracted_img)
        cv2.imshow("Extracted images", res_display_extracted_img)

    if DISPLAY_ANNOTATED_IMGS:
        annotated_img = annotate_image(img, grid_contours, grouped_items, roi)

        # Resize image for display
        res_annotated_img = resize_image(annotated_img)

        # Display the screenshot with annotations
        cv2.imshow("Display annotations", res_annotated_img)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()


def main():
    print("Make sure you are connected to the ADB, check `adb devices`!\n")
    # Starting adb daemon server
    os.system("adb start-server")
    time.sleep(1)

    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    
    # Create an Termux:GUI connection
    if c.RUN_ON_MOBILE:
        connection = tg.Connection()

    # If no device is detected, open the developer options
    if len(devices) == 0:
        if c.RUN_ON_MOBILE:
            connection.toast("Please connect to ADB Wi-Fi IP from developer options", long = True)
            os.system("am start -a com.android.settings.APPLICATION_DEVELOPMENT_SETTINGS")
        print("No device found. Please connect to device using ADB!")
        return

    device = devices[0]

    print("Checking if the Aliexpress app is running")

    # Check if Aliexpress app is running
    wait_for_ali_app(device)

    img = get_screen_capture(device)

    # Only try to merge objects with a similarity above this threshold
    height, width, _ = img.shape
    
    # Display control overlay on mobile and create an thread to verify input
    if c.RUN_ON_MOBILE:
        play_pause_id, farm_id, exit_id = display_overlay_on_android(height, connection)
        watcher = threading.Thread(
                            target=action_on_overlay_button_press, 
                            args=(connection, play_pause_id, farm_id, exit_id), 
                            daemon=True
                  )
        watcher.start()

    # Define the region of interest for duplicate findings
    # Top, bottom, left, right padding
    roi = int(c.ROI_TOP * height), int(c.ROI_BOTTOM * height), int(width * c.ROI_PADDING)

    # Generate ROI grid contours
    grid_contours = generate_grid_contours(img, roi, c.GRID_PADDING)

    # Remember the energy farm status
    farm_the_energy = c.AUTO_FARM_ENERGY

    while True:
        do_button_flags(img, device)
            
        img = get_screen_capture(device)

        extracted_imgs, count_blanks = extract_imgs_from_contours(img, grid_contours)

        grouped_items = group_similar_imgs(extracted_imgs, c.SIMILARITY_THRESHOLD)

        check_if_should_exit(device)

        swipe_elements(device, grid_contours, grouped_items, roi)
        
        check_if_space_left(count_blanks, grouped_items)

        # Check the energy left and matches
        if c.CHECK_ENERGY_LEVEL and len(grouped_items) <= c.MAX_GENERATOR_GROUP_NUMBERS:
            if generate_objects(device, width, height, img, count_blanks+len(grouped_items)) == False:
                if farm_the_energy:
                    print("Starting to farm energy.")
                    farm_energy(img, device)
                    print("Finish farming.")
                    farm_the_energy = False
                else:
                    print("No energy to farm. Exit.")
                    break

        debug_display_img(img, grid_contours, grouped_items, roi, extracted_imgs)

        check_if_should_exit(device)

        if c.AUTOMATIC_DELIVERY:
            try_to_delivery(device, img.shape)


if __name__ == "__main__":
    main()