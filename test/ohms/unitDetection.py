from ultralytics import YOLO
from PIL import Image

# model = YOLO()
# model.train(data='ohms/data.yaml', epochs=300)


model = YOLO('train6/weights/best.pt')
results = model.predict("test6.jpg" ,save=True, save_txt=True, imgsz=640) 