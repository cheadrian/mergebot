from ppadb.client import Client as AdbClient
import cv2
import numpy as np
import time
import configuration as c


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

#  Applies image processing techniques, including Gaussian blur and Adaptive Thresholding, Sobel edge detection or simple thresholding.
def apply_processing(img, sob_thresh=35):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract edges using Sobel
    sobel_edges = sobel_edge_detector(gray, sob_thresh)
    return sobel_edges

    return adaptive_threshold


# Creates a grid of rectangular contours within the specified region of interest (ROI) with padding.
def generate_grid_contours(img, roi, padding=5):
    height, width, _ = img.shape
    roi_min, roi_max, width_padding = roi
    max_rows, max_col = 9, 7
    square_size = (roi_max - roi_min) // max_rows
    contours = []

    # Draw rectangles mesh and find contours with padding
    for row in range(max_rows):
        for col in range(max_col):
            x = (col * square_size) + int(width_padding) + padding
            y = roi_min + row * square_size + padding
            contour = np.array(
                [
                    (x, y),
                    (x + square_size - 2 * padding, y),
                    (x + square_size - 2 * padding, y + square_size - 2 * padding),
                    (x, y + square_size - 2 * padding),
                ]
            )
            contours.append(contour)

    return contours


# Extracts images from a list of contours. Applies image processing (Sobel edge detection) and morphological dilation for better blob extraction.
img_dilation_kernel = np.ones((7, 7), np.uint8)


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

    # Count the number of white pixels
    diff_pixels_cnt = np.count_nonzero(diff_img)

    normalized_similarity = 1 - (diff_pixels_cnt / (height * width))

    return normalized_similarity


# Groups similar images based on the specified similarity threshold, ignoring blank and ignored positions. Returns a list of grouped items.
def group_similar_imgs(imgs, compare_threshold=0.8):
    grouped_items = []
    visited = set()

    for i, img1 in enumerate(imgs):
        if img1[1] or i < c.IGNORED_MATCH_POSITIONS:
            visited.add(i)
            continue

        if i not in visited:
            group = [i]
            found_match = False  # Flag to check if any similar blob is found

            for j, img2 in enumerate(imgs):
                if img2[1] or j < c.IGNORED_MATCH_POSITIONS:
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
    img = cv2.circle(img, (go_btn_left, go_btn_top), 20, (125, 25, 255), -1)
    img = cv2.circle(img, (close_btn_left, close_btn_top), 20, (125, 25, 255), -1)

    # Draw line and points for automatic delivery
    img = cv2.line(
        img, (width // 2, delivery_top), (width, delivery_top), (255, 255, 0), 10
    )
    img = cv2.circle(img, (width // 2, delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (int(width // 1.5), delivery_btn_top), 20, (255, 50, 255), -1)
    img = cv2.circle(img, (int(width // 1.2), delivery_btn_top), 20, (255, 50, 255), -1)

    # Draw ignored contours
    for ig in range(c.IGNORED_MATCH_POSITIONS):
        cv2.drawContours(img, [contours[ig]], 0, (0, 0, 255), 4)

    if c.CHECK_ENERGY_LEVEL:
        for pos in c.GENERATOR_POSITIONS:
            cv2.drawContours(img, [contours[pos - 1]], 0, (0, 160, 255), 6)

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
                x1 + w1 // 2, y1 + h1 // 2, x2 + w2 // 2, y2 + h2 // 2, 50
            )

            # Update the set of already swiped positions
            already_swiped_positions.update([position1, position2])

    if len(groups) > 0:
        # Touch the first element of the list after merge, because of contouring
        x1, y1, _, _ = cv2.boundingRect(contours[0])
        device.input_tap(x1, y1)


def try_to_delivery(device, img_shape):
    height, width, _ = img_shape
    delivery_top = int(c.DEL_TOP * height)
    delivery_btn_top = int(c.DEL_BTN_TOP * height)

    # Swipe through the delivery list and try to press the "Delivery" button
    for i in range(6):
        device.input_swipe(width - 100, delivery_top, width // 2, delivery_top, 100)
        device.input_tap(width // 2, delivery_btn_top)
        device.input_tap(int(width // 1.5), delivery_btn_top)
        device.input_tap(int(width // 1.2), delivery_btn_top)

    # Go back
    for i in range(6):
        device.input_swipe(width // 2, delivery_top, width - 100, delivery_top, 100)


#  Generates objects by clicking on specified generator positions.
def generate_objects(device, contours, img):
    energy_left = get_energy_level(img)
    for pos in c.GENERATOR_POSITIONS:
        x, y, _, _ = cv2.boundingRect(contours[pos - 1])
        if energy_left <= c.MIN_ENERGY_LEVEL:
            print("No energy left")
            return False
        device.input_tap(x, y)
        device.input_tap(x, y)
        energy_left = energy_left - 1

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

    # Preprocess the cropped image for better text recognition
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # # Apply thresholding to enhance text visibility
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.floodFill(thresh, None, (5, 5), 0, flags=8)

    # Use pytesseract to extract numeric text from the preprocessed image
    custom_config = r"--oem 3 --psm 7 outputbase digits"  # Tesseract OCR configuration for numeric digits
    text = pytesseract.image_to_string(thresh, config=custom_config)

    if DISPLAY_EXTRACTED_IMGS:
        cv2.imshow("Extracted energy", thresh)
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
        time.sleep(17)
        device.input_keyevent("BACK")
        time.sleep(2)
        # Hit the "Claim" button
        device.input_tap(width * c.GO_LEFT, height * c.GO_TOP)
        time.sleep(3)
    # Exit task menu
    device.input_tap(width * c.EX_LEFT, height * c.EX_TOP)
    time.sleep(1)


# Combines a list of binary images into a grid with the specified number of columns and rows, just for debugging display.
def combine_binary_images(extracted_imgs, columns=7, rows=9):
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


#  Checks if the current app running on the device is Aliexpress and the screen is on.
def check_app_in_foreground(device, target):
    result = device.shell("dumpsys window | grep mCurrentFocus")
    if target in result:
        return True

    return False


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


def main():
    print("Make sure you are connected to the ADB, check `adb devices`!\n")

    time.sleep(1)

    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()

    if len(devices) == 0:
        print("No device found. Please connect to device using ADB!")
        return

    device = devices[0]

    print("Checking if the Aliexpress app is running")

    # Check if Aliexpress app is running
    wait_for_ali_app(device)

    img = get_screen_capture(device)

    # Only try to merge objects with a similarity above this threshold
    height, width, _ = img.shape

    # Define the region of interest for duplicate findings
    # Top, bottom, left, right padding
    roi = int(c.ROI_TOP * height), int(c.ROI_BOTTOM * height), int(width * c.ROI_PADDING)

    # Generate ROI grid contours
    grid_contours = generate_grid_contours(img, roi, c.GRID_PADDING)

    # Remember the energy farm status
    farm_the_energy = c.AUTO_FARM_ENERGY

    while True:
        # Read the screenshot in memory
        img = get_screen_capture(device)

        extracted_imgs, count_blanks = extract_imgs_from_contours(img, grid_contours)
        
        if count_blanks < c.MIN_SPACES_ON_BOARD:
            print("There's no space left on the grid. Bot will exit.")
            break

        grouped_items = group_similar_imgs(extracted_imgs, c.SIMILARITY_THRESHOLD)

        if not check_app_in_foreground(device, c.TARGET_APP_PKG):
            print("Aliexpress app is not running anymore")
            break

        # Check the energy left and matches
        if c.CHECK_ENERGY_LEVEL and len(grouped_items) <= c.MAX_GENERATOR_GROUP_NUMBERS:
            if (
                generate_objects(device, grid_contours, img) == False
                and len(grouped_items) == 0
            ):
                print("No group found.")
                if farm_the_energy:
                    print("Starting to farm energy.")
                    farm_energy(img, device)
                    print("Finish farming.")
                    farm_the_energy = False
                else:
                    print("No energy to farm. Exit.")
                    break

        # These are for debugging and calibration purposes
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
                break
                cv2.destroyAllWindows()

        if not check_app_in_foreground(device, c.TARGET_APP_PKG):
            print("Aliexpress app is not running anymore")
            break

        # Swipe duplicates one over another
        swipe_elements(device, grid_contours, grouped_items, roi)

        # Add an delay after each iteration to let all items to merge
        time.sleep(0.2)

        if c.AUTOMATIC_DELIVERY:
            try_to_delivery(device, img.shape)


if __name__ == "__main__":
    main()
