[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5NorvP5a)
# Markers, Computer Vision and AR
## Quickstart
1. Create a Virtual Environment (recommended)
2. Install required packages
```
pip install -r requirements.txt
```
3. Run the Image Transformation
```
python perspective_transformation/image_extractor.py --input perspective_transformation/sample_image.jpg --output perspective_transformation/output.jpg --width 800 --height 600
```
4. Run the AR Game
```
python ar_game/AR_game.py
```
## Instructions
### Perspective Transformation
1. Click four corner points in the image
2. After 4 points are chosen the transformed result is shown in a new window
3. Press 'S' to save, or 'ESC' to discard and restart
### AR Game: Whack-a-mole
The goal of this game is to hit targets that appear at random positions. The player has to tap a target before it disappears. Make sure that your markers are well lit and visible to the camera.