"""The windowcapture.py is a class meant to take screenshots on a windows OS.\
It is meant to efficiently take as many screenshots in a second (FPS)."""

import numpy as np
import win32gui
import win32ui
import win32con


class WindowCapture:
    """Kudos to @learncodebygaming for this base class to work with.
    His github can be found here: https://github.com/learncodebygaming/opencv_tutorials/tree/master/004_window_capture
    """
    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 10000
    cropped_y = 10000
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name: str):
        '''Initialize border parameters for picture cut off.
        params:  window_name: Name of window to take screen shots of. '''
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)  # Left, Top, Right, Bottom
        self.w = window_rect[2] - window_rect[0]  # Width = Right - Left
        self.h = window_rect[3] - window_rect[1]  # Height = Bottom - Top

        # account for the window border and titlebar and cut them off
        border_pixels = 100
        titlebar_pixels = 450  # Manages the y axis crop.
        self.w = self.w - border_pixels
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):
        """Process of getting screen and taking screenshot of the screen."""
        # get the window image data
        # wDC = win32gui.GetWindowDC(self.hwnd)
        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        dcObj = win32ui.CreateDCFromHandle(hwndDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj,
                   (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    def list_window_names(self):
        """If applicable, you can get the names of all visible \
        screens to use as input in the class."""
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
