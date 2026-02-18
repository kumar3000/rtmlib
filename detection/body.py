import cv2
from rtmlib import Body, draw_skeleton

# import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body = Body(
    pose='rtmo',
    to_openpose=openpose_skeleton,
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device,
)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)
    if frame_idx % 20 == 0:
        print(f"Frame {frame_idx}\nScores: {scores}")

    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.1,
                             line_width=1)

    img_show = cv2.resize(img_show, (1280, 960))
    cv2.imshow('img', img_show)
    if cv2.waitKey(1) == ord('q'):
        break