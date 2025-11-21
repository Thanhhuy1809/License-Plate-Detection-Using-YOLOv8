from ultralytics import YOLO

def main():
    model = YOLO("yolov8x.pt")
    results = model.train(
        data="mydata.yaml",
        epochs=100,
        device=0,
        imgsz=640,
        batch=8,
    )

if __name__ == "__main__":
    main()
