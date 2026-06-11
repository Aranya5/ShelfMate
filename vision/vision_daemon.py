import cv2
import requests
import numpy as np
import json
from ultralytics import YOLO

print("1. Loading YOLOv8 Nano...")
model = YOLO('yolov8n.pt')

BACKEND_URL = "http://localhost:5001/api/shelf-events"

print("2. Loading Test Video Footage...")
cap = cv2.VideoCapture('test.mp4') 

# --- LOAD MULTIPLE SHELVES ---
try:
    with open('zones.json', 'r') as f:
        zone_data = json.load(f)
        # Convert the JSON lists into a list of NumPy arrays
        shelves = [np.array(shelf, np.int32) for shelf in zone_data["shelves"]]
    print(f"✅ Successfully loaded {len(shelves)} digital shelves.")
except FileNotFoundError:
    print("❌ Error: zones.json not found.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Lowered confidence to 0.45 to prevent ID fragmentation
    results = model.track(frame, persist=True, classes=0, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # Draw all shelves on the screen
    for i, shelf_zone in enumerate(shelves):
        cv2.polylines(annotated_frame, [shelf_zone], isClosed=True, color=(255, 0, 0), thickness=2)
        # Add a label so we know which shelf is which
        cv2.putText(annotated_frame, f"Shelf {i+1}", tuple(shelf_zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    active_shopper_ids = []
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            active_shopper_ids.append(track_id)
            feet_x, feet_y = int((x1 + x2) / 2), int(y2)
            cv2.circle(annotated_frame, (feet_x, feet_y), radius=5, color=(0, 255, 0), thickness=-1)

            # --- MULTI-SHELF COLLISION CHECK ---
            for i, shelf_zone in enumerate(shelves):
                is_inside = cv2.pointPolygonTest(shelf_zone, (feet_x, feet_y), measureDist=False)
                if is_inside >= 0:
                    print(f"🎯 Shopper {track_id} is browsing Shelf {i+1}!")
                    cv2.putText(annotated_frame, f"BROWSING S-{i+1}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("ShelfMate AI Vision", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Set to 1 for maximum speed
        break

cap.release()
cv2.destroyAllWindows()