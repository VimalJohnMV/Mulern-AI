from ultralytics import YOLO

def run_detection():
    # 1. Load a pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # 2. Run detection on an image
    # You can use a URL or a local path
    source = 'https://ultralytics.com/images/bus.jpg'
    
    # Run inference
    # 'save=True' automatically saves the result with bounding boxes to 'runs/detect/predict/'
    results = model.predict(source=source, save=True, conf=0.5)

    # 3. Save the result to a specific filename (results.jpg)
    # The first item in results contains our detection data
    for result in results:
        result.save(filename='results.jpg')
        print("Detection complete. Result saved as 'results.jpg'")

if __name__ == "__main__":
    run_detection()
