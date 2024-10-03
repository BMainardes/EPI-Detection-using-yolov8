from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # ou o caminho para o seu modelo YOLO
    model.train(
        data=r'C:\Users\Bruno\Downloads\EPI detection.v12i.yolov8\data.yaml',
        epochs=100,
        batch=8,
        imgsz=640,
        device='cuda'
    )
