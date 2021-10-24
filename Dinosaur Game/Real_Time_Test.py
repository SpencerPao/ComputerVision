"""Uses trained model to play dinosaur game."""
from Window_capture.utils.windowcapture import WindowCapture
from Window_capture.utils.getkeys import keys
from model import BasicNNet
import cv2 as cv
import pickle
import keyboard
"""CHANGE FILE PATH"""
#model = pickle.load(open("Modeling/Existing_Models/log-reg.pkl", 'rb'))  # horrible.
model = torch.load('nn.model')
print("Model Loaded.")

def map_keys_rev(pred_vec):
    """ Take a vector of classifications and return keyboard outputs """
    result = torch.zeros_like(pred_vec)
    key_dict = {0: -1, 1: 38, 2: 40}
    for i in range(pred_vec.shape[0]):
        result[i] = key_dict[pred_vec[i]]
    return result

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
        preds = model(torch.Tensor(screenshot.flatten().reshape(1, 129600)))
        _, result = torch.max(preds.data, 1)
        # result = model.predict(screenshot.flatten().reshape(1, 129600))
        # flatten images then converted to dataframe for easier removal of idx
        print("Prediction is: ", result)
        # https://stackabuse.com/guide-to-pythons-keyboard-module/
        if result[0] == 38:  # Jump!
            keyboard.send('up')
        elif result[0] == 40:  # duck!
            keyboard.send('down')
        else:
            pass
        # End simulation
        key = keys()
        if key == 0x51:
            cv.destroyAllWindows()
            break
    print("Exit out of loop.")


if __name__ == '__main__':
    main()
