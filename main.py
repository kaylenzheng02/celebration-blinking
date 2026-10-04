import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Confetti colors in OpenCV's blue, green, red format.
COLORS = [
    (180, 100, 255),  # Pink
    (255, 210, 100),  # Blue
    (100, 230, 255),  # Yellow
    (170, 255, 130),  # Green
    (255, 140, 200),  # Purple
]

# Iris centers from the MediaPipe face mesh. Used as the confetti origin.
LEFT_IRIS = 468
RIGHT_IRIS = 473

# Both eyes must close, then open again, before the next burst.
BLINK_ON = 0.5
BLINK_OFF = 0.3

particles = []


def create_confetti(x, y, bias_x=0):
    """Pop a burst of confetti at one eye."""
    for _ in range(16):
        particles.append({
            "x": float(x),
            "y": float(y),
            "vx": random.uniform(-200, 200) + bias_x,
            "vy": random.uniform(-460, -80),
            "color": random.choice(COLORS),
            "size": random.randint(4, 10),
            "life": random.uniform(1.2, 2.6),
        })


def draw_confetti(frame, dt):
    """Move and draw particles, then remove expired ones."""
    height, width = frame.shape[:2]

    for particle in particles[:]:
        particle["x"] += particle["vx"] * dt
        particle["y"] += particle["vy"] * dt

        # Gravity makes the confetti fall.
        particle["vy"] += 500 * dt
        particle["life"] -= dt

        if (
            particle["life"] <= 0
            or particle["y"] > height + 20
            or particle["x"] < -20
            or particle["x"] > width + 20
        ):
            particles.remove(particle)
            continue

        x = int(particle["x"])
        y = int(particle["y"])
        size = particle["size"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + size, y + size // 2 + 1),
            particle["color"],
            -1,
        )


def blink_amount(blendshapes):
    """How closed both eyes are, from 0 (open) to 1 (shut)."""
    scores = {
        shape.category_name: shape.score
        for shape in blendshapes
    }
    left = scores.get("eyeBlinkLeft", 0.0)
    right = scores.get("eyeBlinkRight", 0.0)
    return min(left, right)


def eye_centers(landmarks, width, height):
    """Pixel positions of the two eyes."""
    if len(landmarks) > RIGHT_IRIS:
        indexes = (LEFT_IRIS, RIGHT_IRIS)
        return [
            (int(landmarks[i].x * width), int(landmarks[i].y * height))
            for i in indexes
        ]

    pairs = ((33, 133), (263, 362))
    centers = []
    for a, b in pairs:
        x = int((landmarks[a].x + landmarks[b].x) / 2 * width)
        y = int((landmarks[a].y + landmarks[b].y) / 2 * height)
        centers.append((x, y))
    return centers


model_path = Path(__file__).with_name("face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
)

camera = cv2.VideoCapture(0)

try:
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions "
            "and close other apps using it."
        )

    with vision.FaceLandmarker.create_from_options(options) as tracker:
        previous_time = time.monotonic()
        last_timestamp = -1
        eyes_closed = False

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            # Mirror the video so it behaves like a mirror.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            now = time.monotonic()
            dt = min(now - previous_time, 0.05)
            previous_time = now

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp = max(int(now * 1000), last_timestamp + 1)
            last_timestamp = timestamp
            result = tracker.detect_for_video(image, timestamp)

            if result.face_landmarks and result.face_blendshapes:
                closed = blink_amount(result.face_blendshapes[0])

                # One burst per blink, from both eyes.
                if not eyes_closed and closed >= BLINK_ON:
                    eyes_closed = True
                    centers = eye_centers(
                        result.face_landmarks[0], width, height
                    )
                    create_confetti(*centers[0], bias_x=-40)
                    create_confetti(*centers[1], bias_x=40)
                elif eyes_closed and closed <= BLINK_OFF:
                    eyes_closed = False
            else:
                eyes_closed = False

            draw_confetti(frame, dt)

            cv2.putText(
                frame,
                "Blink! | C: clear | Q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Blink Confetti", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                particles.clear()

finally:
    camera.release()
    cv2.destroyAllWindows()
