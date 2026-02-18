import requests
import supervision as sv
import cv2
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Error: No frame!")
        break

    image = Image.open(frame)
    detections = model.predict(image, threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.BoxAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    cv2.imshow("RFDETR TEST", annotated_image)

    if cv2.waitKey(1) == ord('q'):
        break
