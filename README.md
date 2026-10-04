# A Little Celebration

A webcam toy. Look at the camera and blink, and a burst of confetti pops from both eyes.

The window uses your face to notice a blink. Confetti falls with a little gravity, then fades away. One burst happens each time both eyes close and open again.

## What you need

- Python 3.12
- A webcam

## Setup

From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python mediapipe
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "face_landmarker.task"
```

`face_landmarker.task` is the face model. It has to sit next to `main.py`. Pip does not install it.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

Allow the camera if Windows asks.

## Controls

- Blink both eyes to release confetti
- **C** clears the confetti
- **Q** quits
