"""Uses trained model to play dinosaur game."""
from Window_capture.utils.windowcapture import WindowCapture
from Window_capture.utils.getkeys import keys
import cv2 as cv
import pickle
import keyboard
import time
import xgboost as xgb
"""CHANGE FILE PATH"""
# model = pickle.load(open("Modeling/Existing_Models/log-reg.pkl", 'rb'))  # horrible.
# model = pickle.load(open("Modeling/Existing_Models/xgboost_dino_2.pkl", 'rb'))  # horrible.
model = pickle.load(open("Modeling/Existing_Models/xgboost_dino_SMOTE.pkl", 'rb'))  # horrible.

print("Model Loaded.")


def main():
    wincap = WindowCapture('T-Rex Game – Google Dino Run - Google Chrome')
    while True:
        print("Press 'Space' to start")
        key = keys()
        if key == 32:
            print("Starting Program")
            break

    while(True):
        # get an updated image of the game
        screenshot = wincap.get_screenshot()
        # Crop image
        screenshot = screenshot[0:screenshot.shape[0], 250:700, :]
        gray_images = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        screenshot = cv.Canny(gray_images, threshold1=100, threshold2=200)
        # 417600 for full view (comment line 121) 129600
        screenshot = xgb.DMatrix(screenshot.flatten().reshape(1, 129600))
        start_time = time.time()
        # result = model.predict(screenshot.flatten().reshape(1, 129600))
        result = model.predict(screenshot)
        print("Prediction time --- %s seconds ---" %
              (time.time() - start_time), "result: ", result)
        # flatten images then converted to dataframe for easier removal of idx
        # print("Prediction is: ", result)
        # https://stackabuse.com/guide-to-pythons-keyboard-module/
        if result[0] == 1:  # Jump!
            keyboard.send('up')
        elif result[0] == 2:  # duck!
            keyboard.send('down')
        else:
            pass
        # End simulation
        key = keys()
        if key == 0x51:
            cv.destroyAllWindows()
            break

        # if result[0] == 38:  # Jump!
        #     keyboard.send('up')
        # elif result[0] == 40:  # duck!
        #     keyboard.send('down')
        # else:
        #     pass
        # # End simulation
        # key = keys()
        # if key == 0x51:
        #     cv.destroyAllWindows()
        #     break
    print("Exit out of loop.")


if __name__ == '__main__':
    main()
