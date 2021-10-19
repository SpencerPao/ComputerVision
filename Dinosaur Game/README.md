# << IN PROGRESS >>
---
# YouTube Videos associated with this entire Codebase:
- [YouTube video link Part 1 (Capturing Image Data)](https://youtu.be/6iekqFLAxl0)
- [YouTube video link Part 2 (Cleaning image Data and Live Testing with Basic Model)](https://youtu.be/K9XMAnwO7wM)

---
## Computer Vision Model for beating the Dinosaur Game


Source code for the OpenCV Object Detection in Games series on the **Learn Code By Gaming** YouTube channel.
- [**Heavy Inspiration from the following codebase!**](https://github.com/learncodebygaming/opencv_tutorials)
- [**His Nifty tutorials can be found here on YouTube**](https://www.youtube.com/playlist?list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI)

---

### Major Thanks to Zorian for helping set up the Modeling Notebook. 
His work can be found in the Modeling/basic-models.ipynb

You can check him out: 
  - [Zorian's YouTube Channel](https://www.youtube.com/channel/UC0oMmMPgGVqnDqNTyAIqTpw)
  - [Zorian's Github](https://zorian15.GitHub.io)

---
## The Link to the Dinosaur Game can be found [**Here**](https://trex-runner.com/)
![Dinosaur_Image_Screenshot](Dinosaur_Screenshot.PNG) 
## Requirements
- Linux & Windows OS (Mac users might have to use a google collab notebook)
- run **pip install -r requirements.txt** in shell. (To just run models)

## Real_Time_Test.py
- Use script once model has been trained.
- Runs trained model in the **Existing/Models/** folder
- run **python Real_Time_test.py** to see if trained model beats various obstacles and maybe even the game!

### Window Capture Folder
- Tools for taking a screen shot for the dinosaur game. I will be using this to store the appropriate PNG files into its associated folders (Jump, Duck).

### Dinosaur_Game_Data_Exploration.ipynb 
- Playing around with Windows OS and checking to see if functions work. 

### Future Line of Work
- Study which Computer vision model is most appropriate to beating the game at runtime
- Train Model on Screenshots from the Window Capture Folder
- Integrate with System (as a microservice)
- Potentially deploy for others to use

