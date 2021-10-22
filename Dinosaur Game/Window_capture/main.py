"""Main execution for generating screenshot data of dinosaur game."""
from utils.getkeys import keys
import cv2 as cv
import win32con as con
import numpy as np
import os
from time import time, sleep
from utils.windowcapture import WindowCapture
from typing import List
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = "Data/screenshots.npy"
file_name2 = "Data/command_keys.npy"


def get_data() -> None:
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


def save_data(image_data: np.ndarray, targets: np.ndarray) -> None:
    """Function obtained from ClarityCoders: https://github.com/ClarityCoders/Fall-Guys-AI/blob/master/CreateData.py"""
    np.save(file_name, image_data)
    np.save(file_name2, targets)


def npy_2_greyscale(data: np.ndarray) -> List[int]:
    """Gray scales screen shot images from BGR->Gray"""
    gray = []
    for img in data:
        gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return gray


def canny_images(data: np.ndarray) -> List[int]:
    """Gray scales screen shot images from BGR->Gray
    data: Gray scaled images."""
    canny_arr = []
    for img in data:
        canny_arr.append(cv.Canny(img, threshold1=100, threshold2=200))
    return canny_arr


def remove_faults(target: List[int]) -> int:
    """Remove last moments of failure from target."""
    targ = target
    to_remove = 0  # value to remove before last -1
    v = 0  # 2nd value to check to break loop
    ta = False  # Target acquired.
    for t in targ[::-1]:
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
    return to_remove


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
            target.append(key)
            sleep(0.2)  # decided to do 0.2 second wait.
        elif key == con.VK_DOWN:
            print("Down Arrow Key Initiated.")
            image_data.append(screenshot)
            target.append(key)
        else:
            target.append(-1)
            image_data.append(screenshot)
    print("Done")

    to_remove = remove_faults(target)
    '''Removing the error triggers and residual screenshots.'''
    image_data = image_data[:-to_remove]
    target = target[:-to_remove]
    '''Comment If else statement below if you want to capture no action screenshots
    Capturing elements that will be thrown out (in this case -1)
    IF you don't want non - actions to be captured.'''
    if len(target) > 0:
        res_list = [i for i, value in enumerate(target) if value == -1]
        print("Grey Scaling...")
        gray_images = npy_2_greyscale(image_data)
        print("Canny Edge Detection...")
        c_imgs = np.asarray(canny_images(gray_images))
        images_flat = pd.DataFrame(c_imgs[:, :, :].flatten().reshape(c_imgs.shape[0], 417600))
        # flatten images then converted to dataframe for easier removal of idx
        images_flat = images_flat.drop(images_flat.index[res_list])
        target = np.delete(target, res_list)
        # Save data.
        print(len(target), images_flat.shape)
        print(np.unique(target, return_counts=True))
        save_data(image_data, target)
        print("Finished Saving image data and target values.")
    else:
        print("No Data to save...")

    '''Uncomment this section if you want no-action records. \
    Make sure to comment out the above if else statement.'''
    # if len(target) > 0:
    #   print("Grey Scaling...")
    #   gray_images = npy_2_greyscale(image_data)
    #   print("Canny Edge Detection...")
    #   c_imgs = np.asarray(canny_images(gray_images))
    #   images_flat = pd.DataFrame(c_imgs[:, :, :].flatten().reshape(c_imgs.shape[0], 417600))
    # flatten images then converted to dataframe for easier removal of idx
    # # Save data.
    #   print(len(target), images_flat.shape)
    #   print(np.unique(target, return_counts=True))
    #   save_data(image_data, target)
    # else:
    #   print("No Data to save...")


if __name__ == '__main__':
    main()
