import time
import cv2
from rtmlib import Hand, PoseTracker, draw_skeleton

# import numpy as np

device = "cpu"
backend = "onnxruntime"  # opencv, onnxruntime, openvino

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

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    # s = time.time()
    keypoints, scores = hand(frame)
    # print(f"Frame: {frame_idx}\n{scores}")
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}\nScores: {scores}")

    img_show = frame.copy()
    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.1,
        line_width=1,
    )

    img_show = cv2.resize(img_show, (1280, 960))
    cv2.imshow("img", img_show)
    if cv2.waitKey(1) == ord("q"):
        break
