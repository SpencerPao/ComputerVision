"""Grayscsale Screenshot & Canny Edge Detection."""
import numpy as np
import cv2 as cv
import os
from pathlib import Path
from typing import List


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


def main():
    """Change the target_address and targets address file names."""
    print("Loading in command keys and screenshot data...")
    target_address = os.path.join(Path(os.getcwd()).parent,
                                  'Window_capture\\Data\\command_keys.npy')
    targets = np.load(target_address)
    print("Command Keys Shape: ", targets.shape)
    screenshot_address = os.path.join(Path(os.getcwd()).parent,
                                      'Window_capture\\Data\\screenshots.npy')
    screenshot_data = np.load(screenshot_address, allow_pickle=True)
    print("Screenshot Shape: ", screenshot_data.shape)
    print("Data load was successful.")
    print("Command Key Frequency", np.unique(targets, return_counts=True))
    gray_images = npy_2_greyscale(screenshot_data)
    c_imgs = canny_images(gray_images)
    np.save('Data/cleaned_data.npy', c_imgs)


if __name__ == '__main__':
    main()
    print("Cleaned Data successfully.")
    print("Modeling Efforts can now be done in Dinosaur Game / Modeling")
