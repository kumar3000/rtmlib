import cv2
from rtmlib import Hand, PoseTracker, draw_skeleton
import numpy as np
import sounddevice as sd
from .sound import real_time

device = "cpu"
backend = "onnxruntime"  # opencv, onnxruntime, openvino

MAX_DISTANCE = 480 # size of pixel distance to achieve max activation

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

hand = PoseTracker(
    Hand,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode="lightweight",  # balanced, performance, lightweight
    backend=backend,
    device=device,
    tracking=False,
)

frame_idx = 0

# Configuration
FS = 44100  # Sampling rate, Hz
FREQUENCY = 440.0 # Sine frequency, Hz
VOLUME = 0.5 # Range [0.0, 1.0]
sine_generator = real_time.generate_sine_wave(FREQUENCY, FS, None)

# --- Activation Logic ---
print("Activating real-time sine wave audio (press Enter to stop)...")

# Initialize the generator
sine_generator = real_time.generate_sine_wave(FREQUENCY, FS, None)

# Open a non-blocking stream
with sd.OutputStream(samplerate=FS, channels=1, callback=real_time.callback):
    # The stream runs in the background. The main program waits for user input.
    input()

print("Sound stopped.")

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        print("Error: No frame")
        break

    keypoints, scores = hand(frame)

    # need to do some euclidean distance calculation wit the two points
    point_1  = keypoints[0][8] # index finger tip
    point_2 = keypoints[0][4] # thumb tip
    distance = np.linalg.norm(point_1 - point_2)
    activation = distance / MAX_DISTANCE
    if activation > 1:
        activation = 1
    # keypoints[0][8] = keypoints[0][8] - 100 # see which keypoints are which
    # print(f"Frame: {frame_idx}\n{scores}")

    if frame_idx % 20 == 0:
        print(f"Frame {frame_idx}\nScores: {scores[0][0]}")
        print(f"Distance: {distance:.3f}")
        print(f"Activation: {activation:.3f}")

    img_show = frame
    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.2,
        line_width=1,
    )
    cv2.putText(img_show, f"Activation: {activation:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255 * activation, 255 * (1 - activation)), 2)

    img_show = cv2.resize(img_show, (1280, 960))
    cv2.imshow("RTMLib Index-Thumb Activation TEST 2/17/2026", img_show)
    if cv2.waitKey(1) == ord("q"):
        break
