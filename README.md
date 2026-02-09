# Ball Tracking System

A computer vision system for tracking balls on a snooker table from video.

For videos to show up create a `videos/` file directory in the root directory of the program.


## Features

- Automatic table corner detection (works with any camera angle)
- Automatic orientation detection (portrait/landscape)
- Perspective transformation to top-down view
- Ball detection and centroid tracking
- Side-by-side or stacked display of original and warped views

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy

## Installation

Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage
Make a `videos` folder and place your videos in the `videos/` folder, then run:
```bash
python main.py
```

Select a video when prompted (by number or file path).

## Pre-commit

This repo includes a basic pre-commit setup for linting/formatting. To enable it:
```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```

## Controls

- SPACE: Pause/Resume
- ESC: Quit
- R: Restart video

## File Structure

- `main.py` - Main entry point
- `detection.py` - Table and ball detection
- `warping.py` - Perspective transformation
- `tracking.py` - Centroid tracking
- `display.py` - Display and visualization
- `videos/` - Video files directory
