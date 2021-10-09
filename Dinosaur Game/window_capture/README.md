# Data Engineering Process for Dinosaur Game
- Extracts screenshots based on keyboard input within game.

# Requirements:
- Ran on Windows Machine (Possibly can be run on Google Collab)
- Requirements.txt --> Create new environment and run **pip install -r requirements.txt** in shell.
- When running main.py, ensure that the T-Rex Game is on Separate screen. NOTE: Pixels might need to be rearranged depending on resolution of screen. My screen is 1920 x 1080.

# Numpy Arrays are too large to be stored on Github
- When you run the program, numpy arrays will populate in the data folder.
- Data/screenshots.npy: This is the screenshots array.
- Data/command_keys.npy: Target Value (32: space, 38: Up Arrow Key, 40: Down Arrow Key)
