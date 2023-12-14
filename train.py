
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') #carregando um modelo pré-treinado
results = model.train(data='config.yaml', epochs=100, imgsz=640)
