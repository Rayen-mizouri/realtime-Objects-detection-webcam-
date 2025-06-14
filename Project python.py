from ultralytics import YOLO  # Import the YOLO model from Ultralytics
import cv2

# Load YOLOv8 model (n = nano version)
model = YOLO('yolov8n.pt')

def detect_object(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)           # Perform detection
        detections = results[0].boxes   # Get detected boxes

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Box coordinates
            conf = box.conf[0].item()                        # Confidence score
            label = int(box.cls[0].item())                   # Class ID
            label_name = model.names[label]                  # Class name

            # Set color based on confidence
            if conf > 0.75:
                color = (0, 255, 0)      # Green (high confidence)
            elif conf > 0.5:
                color = (0, 255, 255)    # Yellow (medium confidence)
            else:
                color = (0, 0, 255)      # Red (low confidence)

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label_name}: {conf * 100:.2f}%"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection function
detect_object(model)
