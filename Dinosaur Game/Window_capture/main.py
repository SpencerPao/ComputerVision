"""Main execution for generating screenshot data of dinosaur game."""
from utils.getkeys import keys
import cv2 as cv
import win32con as con
import numpy as np
import os
from time import time, sleep
from utils.windowcapture import WindowCapture

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = "Data/screenshots_2.npy"
file_name2 = "Data/command_keys_2.npy"


def get_data():
    """Function obtained from ClarityCoders:
    https://github.com/ClarityCoders/Fall-Guys-AI/blob/master/CreateData.py"""
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        image_data = list(np.load(file_name, allow_pickle=True))
        targets = list(np.load(file_name2, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        image_data = []
        targets = []
    return image_data, targets


def save_data(image_data, targets):
    """Function obtained from ClarityCoders: https://github.com/ClarityCoders/Fall-Guys-AI/blob/master/CreateData.py"""
    np.save(file_name, image_data)
    np.save(file_name2, targets)


def main():
    """Houses main function."""
    image_data, target = get_data()
    wincap = WindowCapture('T-Rex Game – Google Dino Run - Google Chrome')
    while True:
        print("Press 'Space' to start -- else 'q' to quit program.")
        key = keys()
        if key == 0x51:  # quit
            cv.destroyAllWindows()
            break
        if key == con.VK_SPACE:
            print("Starting Program")
            break
    loop_time = time()
    while(True):
        # Get screenshot
        screenshot = wincap.get_screenshot()
        print("FPS {}".format(1 / (time() - loop_time)))
        loop_time = time()
        key = keys()
        if key == 0x51:
            cv.destroyAllWindows()
            break
        if key == con.VK_UP:
            print("Up Arrow Key")
            image_data.append(screenshot)
            target.append(key)  # Wait for 0.658 seconds.
            sleep(0.2)
            # Make sure to jump only using the up arrow key
            # Since we are using the space bar key for the start of the program.
            # Once up arrow key goes up -> we want to record screenshot
            # and record key.
        elif key == con.VK_DOWN:
            print("Down Arrow Key Initiated.")
            image_data.append(screenshot)
            target.append(key)
            # Attach screen shot
            # Attach record key.
        else:
            target.append(-1)
            image_data.append(screenshot)
    print("Done")
    to_remove = 0  # value to remove before last -1
    v = 0  # 2nd value to check to break loop
    ta = False  # Target acquired.
    for t in target[::-1]:
        if t == -1:
            to_remove += 1
        elif t != -1 and not ta:
            ta = True
            v = t
            to_remove += 1
        elif ta and t == v:
            to_remove += 1
        elif ta and t != v:
            break

    image_data = image_data[:-to_remove]
    target = target[:-to_remove]
    # Save data.
    save_data(image_data, target)
    print("Finished Saving image data and target values.")


if __name__ == '__main__':
    main()
