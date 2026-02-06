from ultralytics import YOLO

model = YOLO("yolov8n.pt")

FRUIT_CLASSES = ["apple", "banana", "orange", "tomato"]

def detect_fruit(image_path):
    results = model(image_path, conf=0.4)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])


            if label in FRUIT_CLASSES:
                return label, confidence

    return None, 0.0
