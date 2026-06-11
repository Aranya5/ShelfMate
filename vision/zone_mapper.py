import cv2
import json

shelves = []
current_points = []
is_drawing_outer = True # Toggle to track which zone we are drawing

def draw_polygon(event, x, y, flags, param):
    global current_points, shelves, is_drawing_outer
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        
        # Yellow for Outer Consideration, Red for Inner Interaction
        color = (0, 255, 255) if is_drawing_outer else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        
        if len(current_points) > 1:
            cv2.line(frame, tuple(current_points[-2]), tuple(current_points[-1]), color, 2)
            
        if len(current_points) == 4:
            cv2.line(frame, tuple(current_points[-1]), tuple(current_points[0]), color, 2)
            
            if is_drawing_outer:
                # Save the outer zone and prep to receive the inner zone
                shelves.append({"outer": current_points, "inner": []})
                is_drawing_outer = False
                print("🟡 Outer Zone saved! Now click 4 points for the tight INNER Interaction Zone.")
            else:
                # Save the inner zone to the current shelf
                shelves[-1]["inner"] = current_points
                is_drawing_outer = True
                print("🔴 Inner Zone saved! Shelf complete. Map next Outer Zone or press 'q' to quit.")
                
                # Save to JSON format
                with open('zones.json', 'w') as f:
                    json.dump({"shelves": shelves}, f, indent=4)
            
            current_points = [] # Reset mouse clicks
            
        cv2.imshow("Dual-Zone Mapper", frame)

cap = cv2.VideoCapture('test.mp4')
success, frame = cap.read()

print("🛠️ DUAL-ZONE MAPPER ACTIVE")
print("1. Click 4 points on the floor to draw the broad YELLOW Consideration Zone.")

cv2.imshow("Dual-Zone Mapper", frame)
cv2.setMouseCallback("Dual-Zone Mapper", draw_polygon)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()