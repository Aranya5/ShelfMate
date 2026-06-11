import cv2
import json
import numpy as np

shelves = []
current_points = []

def draw_polygon(event, x, y, flags, param):
    global current_points, shelves
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        if len(current_points) > 1:
            cv2.line(frame, tuple(current_points[-2]), tuple(current_points[-1]), (0, 255, 0), 2)
            
        if len(current_points) == 4:
            cv2.line(frame, tuple(current_points[-1]), tuple(current_points[0]), (0, 255, 0), 2)
            
            # Save the completed shelf to our master list
            shelves.append(current_points)
            current_points = [] # Reset for the next shelf
            
            # Save to JSON dynamically
            with open('zones.json', 'w') as f:
                json.dump({"shelves": shelves}, f, indent=4)
                
            print(f"✅ Shelf {len(shelves)} saved! Keep clicking to map another, or press 'q' to quit.")
            
        cv2.imshow("Multi-Zone Mapper", frame)

cap = cv2.VideoCapture('test.mp4')
success, frame = cap.read()

print("🛠️ MULTI-ZONE MAPPER ACTIVE")
print("Click 4 corners to map Shelf 1. Then click 4 more for Shelf 2, etc.")

cv2.imshow("Multi-Zone Mapper", frame)
cv2.setMouseCallback("Multi-Zone Mapper", draw_polygon)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()