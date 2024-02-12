from ultralytics import YOLO
coco_model = YOLO('yolov8n.pt')

license_plate_detector = YOLO()