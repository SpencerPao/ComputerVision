import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
# from WebInterface import WebInterface
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys


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
        image_b64 = self._driver.get_screenshot_as_base64()
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        return screen[..., :3]

    def press_up(self):
        # one space to start game.
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        # one space to start game.
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

    def press_space(self):
        # one space to start game.
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)


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
        self.action_dict[action]()
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

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
