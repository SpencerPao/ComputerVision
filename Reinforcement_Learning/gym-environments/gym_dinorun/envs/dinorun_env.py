''' A HUGE chunk of the codebase came from the following blog:
    https://www.advisori.de/news-projekte/details/a-headless-gym-enviroment-for-every-browser-game/
    So, do make sure to check them out!!!
'''
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
import cv2


class WebInterface:
    def __init__(self,
                 custom_config=True,
                 game_url='chrome://dino',
                 headless=False,
                 chrome_driver_path="chrome_driver/chromedriver"):
        self.game_url = game_url
        self._service = Service(chrome_driver_path)
        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("disable-infobars")
        # Cause we don't want to hear the 100 milestone and jump actions
        _chrome_options.add_argument("--mute-audio")
        # required for windows OS- probably should comment out for any other OS
        _chrome_options.add_argument("--disable-gpu")
        # _chrome_options.add_argument("--start-maximized")
        # _chrome_options.add_argument("--window-size=100,500")
        _chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # if headless:
        #     display = Display(visible=0, size=(1024, 768))
        #     display.start()

        self._driver = webdriver.Chrome(service=self._service,
                                        options=_chrome_options)

#         self._driver.set_window_position(x=-10,y=0)
        try:
            self._driver.get('chrome://dino')
        except WebDriverException:
            pass  # For some reason I get an exception?

    def end(self):
        self._driver.close()

    def grab_screen(self):
        """
            Returns screenshot from the environment.
        """
        image_b64 = self._driver.get_screenshot_as_base64()
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        # Height x width
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
        cropped_screen = screen[int(screen.shape[0]/5):int(3*(screen.shape[0])/5),
                                0:int(screen.shape[1]/3)]
        cropped_screen = cv2.resize(cropped_screen, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        cropped_screen = np.expand_dims(cropped_screen, axis=0)  # Consider observation.
        return cropped_screen

    def press_up(self):
        """
            Execute Jump command for dinosaur.
        """
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        """
            Execute Duck command for dinosaur.
        """
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)


class DinoRunEnv (gym.Env, WebInterface):
    def __init__(self, *args, **kwargs):
        gym.Env.__init__(self)
        WebInterface.__init__(self, *args, game_url='chrome://dino', **kwargs)
        self._driver.execute_script("Runner.config.ACCELERATION=0")

        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self._driver.execute_script(init_script)

        self.action_dict = {0: lambda: None,
                            1: self.press_up,
                            2: self.press_down
                            }
        self.action_space = spaces.discrete.Discrete(3)
        self.reward_range = (-1, 0.1)

    def reset(self):
        '''Resets environment and returns initial observation.'''
        self._driver.execute_script("Runner.instance_.restart()")
        self.step(1)
        time.sleep(2)
        return self.grab_screen()

    def step(self, action):
        ''' Runs one timestep of the game.
            return next state, a reward, and a boolean
        '''
        assert action in self.action_space
        self.action_dict[action]()  # returns some function for every step.
        return self.get_info()

    def get_info(self):
        screen = self.grab_screen()
        score = self.get_score()
        done, reward = (True, -1) if self.get_crashed() else (False, 0.1)
        return screen, reward, score, done

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def render(self):  # Gets a frame (useful for visualization -- to do in future)
        pass

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
