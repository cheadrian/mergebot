import termuxgui as tg
import time
import io
import os
import cv2
import numpy as np
import json
import configuration as c


SCREENSHOT_NAME = "screenshot.jpg"
SCREENSHOT_ANNON = "screenshot_annon.jpg"


# Annotates relevant data to the image.
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

    max_rows, max_col = 9, 7
    padding = c.GRID_PADDING
    square_size = (roi_max - roi_min) // max_rows
    contours = []

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

    # Draw grid
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (255, 0, 0), 2)

    # Draw ignored contours
    for ig in range(c.IGNORED_MATCH_POSITIONS):
        cv2.drawContours(img, [contours[ig]], 0, (0, 0, 255), 4)

    # Draw generator positions
    for pos in c.GENERATOR_POSITIONS:
        cv2.drawContours(img, [contours[pos - 1]], 0, (0, 255, 0), 8)

    return img


# A function that generates a button with specified text, layout, and optional width.
def create_button(activity, text, layout, width=0):
    button = tg.Button(activity, text, layout)
    button.settextsize(16)
    button.setlinearlayoutparams(1)
    if width:
        button.setwidth(width)

    return button


# Processes an image from a given path by annotating it using OpenCV, saves the annotated image, and loads it.
def annotate_and_load_img(screenshot_path, image_viewer):
    img = cv2.imread(screenshot_path)
    annotated_img = annotate_image(img)
    screenshot_anon_path = os.path.expanduser(f"~/{SCREENSHOT_ANNON}")
    cv2.imwrite(screenshot_anon_path, annotated_img)
    with io.open(screenshot_anon_path, "rb") as f:
        image = f.read()
        image_viewer.setimage(image)


def main():
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
            "Select an screenshot of the game to tune parameters like ROI, padding, generator positions for your game.",
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

        paramGridLayout = tg.GridLayout(activity, 2, 3, scrollLinear)
        ingore_matches_txt = tg.TextView(activity, "Ignored positions", paramGridLayout)
        ingore_matches_txt.setgridlayoutparams(0, 0)
        ingore_matches_txt.setwidth(145)
        generator_matches_txt = tg.TextView(
            activity, "Generator positions", paramGridLayout
        )
        generator_matches_txt.setgridlayoutparams(0, 1)
        generator_matches_txt.setwidth(145)

        ignored_matches = tg.EditText(
            activity,
            str(c.IGNORED_MATCH_POSITIONS),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        ignored_matches.setgridlayoutparams(1, 0)
        ignored_matches.setwidth(145)
        generators_matches = tg.EditText(
            activity,
            ",".join(map(str, c.GENERATOR_POSITIONS)),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        generators_matches.setgridlayoutparams(1, 1)
        generators_matches.setwidth(145)

        set_param_btn = create_button(activity, "Set", paramGridLayout)
        set_param_btn.setgridlayoutparams(0, 2, 2, 1)

        energyHorizontalScroll = tg.HorizontalScrollView(activity, scrollLinear)
        energyLinearHorizontal = tg.LinearLayout(
            activity, energyHorizontalScroll, False
        )

        _ = create_button(activity, "ROI", energyLinearHorizontal)
        top_pl_roi_btn = create_button(activity, "↑+", energyLinearHorizontal)
        top_mn_roi_btn = create_button(activity, "↑-", energyLinearHorizontal)
        bot_pl_roi_btn = create_button(activity, "↓+", energyLinearHorizontal)
        bot_mn_roi_btn = create_button(activity, "↓-", energyLinearHorizontal)
        left_pl_roi_btn = create_button(activity, "↔+", energyLinearHorizontal)
        left_mn_roi_btn = create_button(activity, "↔-", energyLinearHorizontal)
        grid_pl_roi_btn = create_button(activity, "G+", energyLinearHorizontal)
        grid_mn_roi_btn = create_button(activity, "G-", energyLinearHorizontal)

        _ = create_button(activity, "ENERGY", energyLinearHorizontal)
        top_mn_eng_btn = create_button(activity, "↑", energyLinearHorizontal)
        top_pl_eng_btn = create_button(activity, "↓", energyLinearHorizontal)
        left_mn_eng_btn = create_button(activity, "←", energyLinearHorizontal)
        left_pl_eng_btn = create_button(activity, "→", energyLinearHorizontal)

        _ = create_button(activity, "GO", energyLinearHorizontal)
        top_pl_go_btn = create_button(activity, "↑", energyLinearHorizontal)
        top_mn_go_btn = create_button(activity, "↓", energyLinearHorizontal)
        left_pl_go_btn = create_button(activity, "←", energyLinearHorizontal)
        left_mn_go_btn = create_button(activity, "→", energyLinearHorizontal)

        _ = create_button(activity, "X", energyLinearHorizontal)
        top_pl_ex_btn = create_button(activity, "↑", energyLinearHorizontal)
        top_mn_ex_btn = create_button(activity, "↓", energyLinearHorizontal)
        left_pl_ex_btn = create_button(activity, "←", energyLinearHorizontal)
        left_mn_ex_btn = create_button(activity, "→", energyLinearHorizontal)
        
        _ = create_button(activity, "D. SWIPE", energyLinearHorizontal)
        top_pl_delswp_btn = create_button(activity, "↑", energyLinearHorizontal)
        top_mn_delswp_btn = create_button(activity, "↓", energyLinearHorizontal)
        
        _ = create_button(activity, "D. BTN", energyLinearHorizontal)
        top_pl_delb_btn = create_button(activity, "↑", energyLinearHorizontal)
        top_mn_delb_btn = create_button(activity, "↓", energyLinearHorizontal)

        _, rootHeight = rootLinear.getdimensions()
        image_viewer = tg.ImageView(activity, scrollLinear)
        image_viewer.setlinearlayoutparams(0)
        image_viewer.setheight(rootHeight - int(rootHeight / 7), True)

        paramGridLayout = tg.GridLayout(activity, 4, 2, scrollLinear)
        min_eng_lvl_txt = tg.TextView(activity, "Minimum energy level", paramGridLayout)
        min_eng_lvl_txt.setgridlayoutparams(0, 0)
        min_eng_lvl_txt.setwidth(145)
        max_farm_act_txt = tg.TextView(
            activity, "Maximum farm actions", paramGridLayout
        )
        max_farm_act_txt.setgridlayoutparams(0, 1)
        max_farm_act_txt.setwidth(145)

        min_eng_lvl = tg.EditText(
            activity,
            str(c.MIN_ENERGY_LEVEL),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        min_eng_lvl.setgridlayoutparams(1, 0)
        min_eng_lvl.setwidth(145)
        max_farm_act = tg.EditText(
            activity,
            str(c.MAX_FARM_SESSIONS),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        max_farm_act.setgridlayoutparams(1, 1)
        max_farm_act.setwidth(145)

        sim_thresh_txt = tg.TextView(activity, "Similarity threshold", paramGridLayout)
        sim_thresh_txt.setgridlayoutparams(2, 0)
        sim_thresh_txt.setwidth(145)
        gen_min_groups_txt = tg.TextView(
            activity, "Press generator when minimum groups number", paramGridLayout
        )
        gen_min_groups_txt.setgridlayoutparams(2, 1)
        gen_min_groups_txt.setwidth(145)

        sim_thresh = tg.EditText(
            activity,
            str(c.SIMILARITY_THRESHOLD),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        sim_thresh.setgridlayoutparams(3, 0)
        sim_thresh.setwidth(145)
        gen_min_groups = tg.EditText(
            activity,
            str(c.MAX_GENERATOR_GROUP_NUMBERS),
            paramGridLayout,
            singleline=True,
            inputtype="number",
        )
        gen_min_groups.setgridlayoutparams(3, 1)
        gen_min_groups.setwidth(145)

        save_btn = create_button(activity, "Save settings", scrollLinear)

        screenshot_path = os.path.expanduser(f"~/{SCREENSHOT_NAME}")
        for event in connection.events():
            if event.type == tg.Event.click and event.value["id"] == screenshot_btn:
                os.system(f"termux-storage-get {screenshot_path}")
            if (
                event.type == tg.Event.click
                and event.value["id"] == load_screenshot_btn
            ):
                with io.open(screenshot_path, "rb") as f:
                    image = f.read()
                    time.sleep(1)
                    image_viewer.setimage(image)
            if event.type == tg.Event.click and event.value["id"] == top_pl_roi_btn:
                c.ROI_TOP += 0.0015
            if event.type == tg.Event.click and event.value["id"] == top_mn_roi_btn:
                c.ROI_TOP -= 0.0015
            if event.type == tg.Event.click and event.value["id"] == bot_pl_roi_btn:
                c.ROI_BOTTOM += 0.0015
            if event.type == tg.Event.click and event.value["id"] == bot_mn_roi_btn:
                c.ROI_BOTTOM -= 0.0015
            if event.type == tg.Event.click and event.value["id"] == left_pl_roi_btn:
                c.ROI_PADDING += 0.001
            if event.type == tg.Event.click and event.value["id"] == left_mn_roi_btn:
                c.ROI_PADDING -= 0.001
            if event.type == tg.Event.click and event.value["id"] == grid_pl_roi_btn:
                c.GRID_PADDING += 1
            if event.type == tg.Event.click and event.value["id"] == grid_mn_roi_btn:
                c.GRID_PADDING -= 1
            if event.type == tg.Event.click and event.value["id"] == top_pl_eng_btn:
                c.ENG_TOP += 0.001
                c.ENG_BOTTOM += 0.001
            if event.type == tg.Event.click and event.value["id"] == top_mn_eng_btn:
                c.ENG_TOP -= 0.001
                c.ENG_BOTTOM -= 0.001
            if event.type == tg.Event.click and event.value["id"] == left_pl_eng_btn:
                c.ENG_LEFT += 0.001
                c.ENG_RIGHT += 0.001
            if event.type == tg.Event.click and event.value["id"] == left_mn_eng_btn:
                c.ENG_LEFT -= 0.001
                c.ENG_RIGHT -= 0.001
            if event.type == tg.Event.click and event.value["id"] == top_mn_go_btn:
                c.GO_TOP += 0.001
            if event.type == tg.Event.click and event.value["id"] == top_pl_go_btn:
                c.GO_TOP -= 0.001
            if event.type == tg.Event.click and event.value["id"] == left_mn_go_btn:
                c.GO_LEFT += 0.001
            if event.type == tg.Event.click and event.value["id"] == left_pl_go_btn:
                c.GO_LEFT -= 0.001
            if event.type == tg.Event.click and event.value["id"] == top_mn_ex_btn:
                c.EX_TOP += 0.001
            if event.type == tg.Event.click and event.value["id"] == top_pl_ex_btn:
                c.EX_TOP -= 0.001
            if event.type == tg.Event.click and event.value["id"] == left_mn_ex_btn:
                c.EX_LEFT += 0.001
            if event.type == tg.Event.click and event.value["id"] == left_pl_ex_btn:
                c.EX_LEFT -= 0.001
            if event.type == tg.Event.click and event.value["id"] == top_pl_delswp_btn:
                c.DEL_TOP -= 0.002
            if event.type == tg.Event.click and event.value["id"] == top_mn_delswp_btn:
                c.DEL_TOP += 0.002                
            if event.type == tg.Event.click and event.value["id"] == top_pl_delb_btn:
                c.DEL_BTN_TOP -= 0.002
            if event.type == tg.Event.click and event.value["id"] == top_mn_delb_btn:
                c.DEL_BTN_TOP += 0.002
            if event.type == tg.Event.click and event.value["id"] == save_btn:
                min_energy_level = int(min_eng_lvl.gettext())
                max_farm_session = int(max_farm_act.gettext())
                similarity_thresh = float(sim_thresh.gettext())
                max_generator_group_numbers = int(gen_min_groups.gettext())
                variables_dict = {
                    "RUN_ON_MOBILE": True,
                    "IGNORED_MATCH_POSITIONS": c.IGNORED_MATCH_POSITIONS,
                    "GENERATOR_POSITIONS": c.GENERATOR_POSITIONS,
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
                    "MIN_ENERGY_LEVEL": min_energy_level,
                    "MAX_FARM_SESSIONS": max_farm_session,
                    "SIMILARITY_THRESHOLD": similarity_thresh,
                    "MAX_GENERATOR_GROUP_NUMBERS": max_generator_group_numbers,
                }
                json_data = json.dumps(variables_dict, indent=4)
                with io.open(c.config_path, "w") as json_file:
                    json_file.write(json_data)
                time.sleep(1)
                exit()
            if event.type == tg.Event.click and event.value["id"] == set_param_btn:
                c.IGNORED_MATCH_POSITIONS = int(ignored_matches.gettext())
                c.GENERATOR_POSITIONS = [
                    int(i) for i in generators_matches.gettext().split(",")
                ]

            if (
                event.type == tg.Event.click
                and event.value["id"] != screenshot_btn
                and event.value["id"] != save_btn
            ):
                annotate_and_load_img(screenshot_path, image_viewer)


if __name__ == "__main__":
    main()
