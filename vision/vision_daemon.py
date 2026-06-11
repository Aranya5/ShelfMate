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

video_fps = cap.get(cv2.CAP_PROP_FPS)
dynamic_delay = int(1000 / video_fps) if video_fps > 0 else 33

# --- SPATIAL MAPPING: Load Explicit Zones ---
try:
    with open('zones.json', 'r') as f:
        zone_data = json.load(f)
        
    shelves = []
    
    for shelf in zone_data["shelves"]:
        # Load both zones directly from your perfectly mapped JSON
        outer_zone = np.array(shelf["outer"], np.int32)
        inner_zone = np.array(shelf["inner"], np.int32)
        
        # Store them as a tuple pairing: (Inner, Outer)
        shelves.append((inner_zone, outer_zone))
        
    print(f"✅ Loaded {len(shelves)} perfectly mapped dual-zone shelves.")
except FileNotFoundError:
    print("❌ Error: zones.json not found. Run zone_mapper.py first!")
    exit()

print("🚀 Vision Daemon Active! (Press 'q' to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- INFERENCE & TRACKING ---
    # Using the custom tracker for memory and lowered confidence to 0.45
    results = model.track(frame, persist=True, tracker="custom_tracker.yaml", classes=0, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # --- DRAW ZONES ---
    # We now iterate directly through the list of tuples instead of using zip()
    for i, (inner, outer) in enumerate(shelves):
        # Draw Outer Halo (Yellow)
        cv2.polylines(annotated_frame, [outer], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        # Draw Inner Zone (Red)
        cv2.polylines(annotated_frame, [inner], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(annotated_frame, f"S-{i+1}", tuple(inner[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    active_shopper_ids = []
    
    # --- BEHAVIOR & COLLISION ANALYSIS ---
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            active_shopper_ids.append(track_id)
            
            # Anchor Point: Bottom-Center (Feet)
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)
            cv2.circle(annotated_frame, (feet_x, feet_y), radius=5, color=(0, 255, 0), thickness=-1)

            # Check Collisions for every shelf
            for i, (inner, outer) in enumerate(shelves):
                # Check Outer Zone First
                in_outer = cv2.pointPolygonTest(outer, (feet_x, feet_y), measureDist=False) >= 0
                
                if in_outer:
                    # If they are in the outer zone, check if they stepped into the Inner Zone
                    in_inner = cv2.pointPolygonTest(inner, (feet_x, feet_y), measureDist=False) >= 0
                    
                    if in_inner:
                        status_text = f"INTERACTING S-{i+1}"
                        color = (0, 0, 255) # Red text
                        print(f"🛒 Shopper {track_id} is INTERACTING with Shelf {i+1}!")
                    else:
                        status_text = f"CONSIDERING S-{i+1}"
                        color = (0, 255, 255) # Yellow text
                        print(f"👀 Shopper {track_id} is considering Shelf {i+1}.")
                        
                    # Draw the status flag above their head
                    cv2.putText(annotated_frame, status_text, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ShelfMate AI Vision", annotated_frame)

    # Telemetry Keep-Alive
    if len(active_shopper_ids) > 0:
        try:
            requests.post(BACKEND_URL, json={"count": len(active_shopper_ids)}, timeout=1)
        except requests.exceptions.ConnectionError:
            pass 

    # Change to 1 if you want maximum speed, or leave as dynamic_delay for natural speed
    if cv2.waitKey(dynamic_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera released safely.")