"""Uses trained model to play dinosaur game."""
from Window_capture.utils.windowcapture import WindowCapture
from Window_capture.utils.getkeys import keys
import cv2 as cv
import pickle
import keyboard

"""CHANGE FILE PATH"""
model = pickle.load(open("\path\to\model.pkl", 'rb'))


def main():
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

    while(True):
        # get an updated image of the game
        screenshot = wincap.get_screenshot()
        result = model.predict(screenshot)
        print("Prediction is: ", result)
        # https://stackabuse.com/guide-to-pythons-keyboard-module/
        if result == 38:  # Jump!
            keyboard.send('up')
        if result == 40:  # duck!
            keyboard.send('down')
        # End simulation
        key = keys()
        if key == "Q":
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
