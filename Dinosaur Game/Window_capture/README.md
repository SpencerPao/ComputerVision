# Data Engineering Process for Dinosaur Game
- Extracts screenshots based on keyboard input within game.

# Requirements:
- Ran on Windows Machine (Possibly can be run on Google Collab)
- Requirements.txt --> Create new environment and run **pip install -r requirements.txt** in shell. (To extract data)
- When running main.py, ensure that the T-Rex Game is on Separate screen. NOTE: Pixels might need to be rearranged depending on resolution of screen. My screen is 1920 x 1080.
- run **python main.py**; press space bar to execute program; "Q" to quit
- Once program is running in background, start playing game. (Keys are only recorded when you hit your up or down arrow key.)

# Numpy Arrays are too large to be stored on Github
- When you run the program, numpy arrays will populate in the data folder.
- Data/screenshots.npy: This is the screenshots array.
- Data/command_keys.npy: Target Value (**-1**: Do Nothing, **38**: Up Arrow Key, **40**: Down Arrow Key): I later associated -1 with 0 and 38 with 1. I did not do the ducking motion in the final model development piece because of redundancy.

# Finished Collecting your data?
- You can now being the modeling portion seen in Dinosaur Game / Window Capture
- The data capturing portion is related to the following [YouTube video](https://youtu.be/6iekqFLAxl0)
- The data cleaning portion is related to the following [YouTube video](https://youtu.be/K9XMAnwO7wM)
- Model evaluation portion can be found [here.](https://www.youtube.com/watch?v=PXQgRGPl4gw)
