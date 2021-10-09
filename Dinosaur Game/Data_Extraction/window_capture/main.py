import cv2 as cv
import win32con as con
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from utils.getkeys import keys
# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = "Data/screenshots.npy"
file_name2 = "Data/command_keys.npy"


def get_data():
    '''Function obtained from ClarityCoders:
    https://github.com/ClarityCoders/Fall-Guys-AI/blob/master/CreateData.py'''
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
    '''Function obtained from ClarityCoders:
    https://github.com/ClarityCoders/Fall-Guys-AI/blob/master/CreateData.py'''
    np.save(file_name, image_data)
    np.save(file_name2, targets)


def main():
    '''Main function to execute.'''
    image_data, target = get_data()
    # initialize the WindowCapture class
    wincap = WindowCapture('T-Rex Game – Google Dino Run - Google Chrome')
    while True:
        print("Press 'Space' to start -- else 'q' to quit program")
        key = keys()
        if key == 'Q':  # quit
            cv.destroyAllWindows()
            break
        if key == 32:
            print("Starting Program")
            break
        elif key == con.VK_UP:
            print("You pressed the up arrow key...")
        elif key == con.VK_DOWN:
            print("You pressed the down arrow key...")

    loop_time = time()
    while(True):
        # get an updated image of the game
        screenshot = wincap.get_screenshot()
        #cv.imshow('Computer Vision', screenshot)
        # Storing the images.
        # debug the loop rate
        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()
        key = keys()
        if key == "Q":
            cv.destroyAllWindows()
            break
        if key == con.VK_UP or key == 32:
            print("Up Arrow key or Space bar initiated")
            #screenshot = cv.resize(screenshot, (224, 224))
            image_data.append(screenshot)
            target.append(key)
        elif key == con.VK_DOWN:
            print("Down Arrow Key initiated")
            #screenshot = cv.resize(screenshot, (224, 224))
            image_data.append(screenshot)
            target.append(key)
    print('Done.')
    save_data(image_data, target)
    print("Finished saving image data with associated target values.")


if __name__ == '__main__':
    main()
