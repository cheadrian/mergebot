

# AliExpress Merge Boss Bot: OpenCV, ADB

## Overview

This repository contains a Python script for an automation bot designed to interact with the AliExpress Merge Boss game. You can run this on PC or using Termux on Android.

Old version can be found on this [GitHub Gist](https://gist.github.com/cheadrian/4331cd8eb95ea6a7097b1830f80db781).

This is the second version of the bot that uses ADB access directly.

Check [this article](https://che-adrian.medium.com/9ac9a6d5581c) to see demo gifs and images.

## Features

Automatically:
- Merge the same items on the grid
- Check energy level
- Farm energy when it's depleted
- Generate new items with selected generators
- Swipe and click delivery buttons
- Stop when there's no energy left
- Stop when user exits the app
- Load user  config or default ones

Additionally adjust:
- Parameters related to buttons and ROI position
- Ignore selected positions
- Matching threshold
- Number of energy farm sessions
- Generators position
- Minimum number of detected groups to generate new items

Display in [real-time](media/run_on_pc.jpg) on the PC, the annotated image and debugging information, like energy level.

## Setup on PC
You need to have [ADB](https://www.xda-developers.com/install-adb-windows-macos-linux/), [Tesseract](https://tesseract-ocr.github.io/tessdoc/Downloads.html) and [Python](https://www.python.org/downloads/) installed and added to your path (commands in Terminal or CMD that should work: python, adb, tesseract, pip) before you can run this script, then connect the phone to the PC using USB cable and activate Android Debugging from developers options on your Android phone. 

On Terminal / CMD:
`adb devices` 
And check if you have on device listed here.

Now you should install the necessary pip packages:
`pip install opencv-python numpy pytesseract pure-python-adb`

## Setup on Termux

First make sure you have installed the [Termux](https://github.com/termux/termux-app/releases), [Termux:API](https://github.com/termux/termux-api/releases) and [Termux:GUI](https://github.com/termux/termux-gui/releases) APKs from git.

If you want a faster mirror then use the `termux-change-repo` before everything else and select a mirror apropriate to your region - using space to select, enter to confirm.

To setup on Termux, copy and paste the code below:

```bash
yes | pkg update -y
yes | pkg upgrade -y
pkg install -y git opencv-python tesseract python android-tools termux-api
pip install numpy pytesseract pure-python-adb termuxgui
git clone https://github.com/cheadrian/mergebot
cd ~/mergebot
```

*Note:* Due to a change in Termux packages the OpenCV-Python has some broken simbols. Check the updates and the problem [description here](https://github.com/cheadrian/mergebot/issues/7).

### ADB pairing on Termux

You should connect to ADB even on Termux, and to do this you can use ADB over WI-FI.

[Here's an video example](https://www.youtube.com/watch?v=BHc7uvX34bM), but don't "adb shell" at end.

Navigate to Settings -> Developer options. Enable USB debugging and enter in "Wireless debugging".

Put that menu in split-screen or PIP with Termux and press the "Pair device with pairing code".

In Termux write (replace IP , PORT, CODE with ones from menu):

```bash
adb pair IP:PORT
PAIRING CODE```

### ADB connection

Now you are paired, you should connect to the ADB.

You will have to do this every time you reopen Termux or reconnect to Wi-Fi. 

In the same menu, you have "IP address & Port". Use these to connect:

```bash
adb connect IP:PORT
adb devices
```

## Configuration

You should adjust configuration based on your device and game status. To do this, on PC you can adjust parameters inside `configuration.py`.

On Termux you can run the configuration script.

Take two screenshots in the game, one with the items to merge, and one on the energy task menu and run the configuration script:

```bash
cd ~/mergebot
python bot_gui.py
```

Adjust the parameters to match the buttons and grid. Should look like in these images:

[Grid, ROI, energy calibration](media/calibration_guide_1.jpg)
[Energy farm calibration](media/calibration_guide_2.jpg)

After that, press save and check the running section.

## Running

Running is simple. 

You should have device connected through ADB first! Check with:

```bash
adb devices
```

There should be one device with "online" status in the list.

After you configured the bot parameters, you can simply:

If you are on Termux, first:

```bash
cd ~/mergebot
```

```bash
python bot_run.py
```

The script is waiting 15 seconds for you to open the AliExpress Merge Boss game. 

It will automatically close when there's no more energy left to farm. 
