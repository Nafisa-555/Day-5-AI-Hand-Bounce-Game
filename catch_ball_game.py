import cv2
import mediapipe as mp
import random

#  Camera 
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
W, H = 640, 480

# MediaPipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Ball factory 
# each ball is a dict to avoid mixing types in a plain list
def make_ball():
    return {
        "x":  float(random.randint(50, W - 50)),
        "y":  float(random.randint(50, 200)),
        "dx": float(random.choice([-3, 3])),
        "dy": float(random.choice([3, 4])),
        "color": (random.randint(100, 255),
                  random.randint(100, 255),
                  random.randint(100, 255)),
        "bouncing": False,   
    }

balls = [make_ball() for _ in range(4)]
radius   = 12
score    = 0
prev_x, prev_y = W // 2, H // 2

# Main loop 
while True:
    success, img = cap.read()
    
    if not success or img is None:
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(3, W)
        cap.set(4, H)
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    bucket_x, bucket_y = prev_x, prev_y

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x1 = int(handLms.landmark[4].x * W)
            y1 = int(handLms.landmark[4].y * H)
            x2 = int(handLms.landmark[8].x * W)
            y2 = int(handLms.landmark[8].y * H)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
           
            bucket_x = prev_x + (cx - prev_x) // 5
            bucket_y = prev_y + (cy - prev_y) // 5
            prev_x, prev_y = bucket_x, bucket_y
            # fingertips
            cv2.circle(img, (x1, y1), 6, (255, 255, 255), -1)
            cv2.circle(img, (x2, y2), 6, (255, 255, 255), -1)

    # bucket
    bucket_w, bucket_h = 100, 40
    bx1 = bucket_x - bucket_w // 2
    by1 = bucket_y - bucket_h // 2
    bx2 = bucket_x + bucket_w // 2
    by2 = bucket_y + bucket_h // 2
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 255), 3)

    # Ball logic 
    for ball in balls:
        ball["x"] += ball["dx"]
        ball["y"] += ball["dy"]
        bx, by = ball["x"], ball["y"]

        # wall bounce
        if bx - radius <= 0 or bx + radius >= W:
            ball["dx"] *= -1
            ball["x"] = max(radius, min(W - radius, bx))

        if by - radius <= 0:
            ball["dy"] *= -1
            ball["y"] = float(radius)

        hit_x = (bx + radius > bx1) and (bx - radius < bx2)
        hit_y = (by + radius > by1) and (by - radius < by2)

        if hit_x and hit_y:
            if not ball["bouncing"]:          
                ball["dy"] *= -1
                ball["y"] = float(by1 - radius)  # push ball above bucket surface
                score += 1
                ball["bouncing"] = True
        else:
            ball["bouncing"] = False          # reset flag once ball leaves bucket

        if ball["y"] > H + radius:
            ball["x"] = float(random.randint(50, W - 50))
            ball["y"] = float(random.randint(50, 180))
            ball["dx"] = float(random.choice([-3, 3]))
            ball["dy"] = float(random.choice([3, 4]))
            ball["bouncing"] = False

        # glow effect
        color = ball["color"]
        for i in range(3, 0, -1):
            cv2.circle(img, (int(ball["x"]), int(ball["y"])),
                       radius + i * 6,
                       (color[0] // (i + 1), color[1] // (i + 1), color[2] // (i + 1)), -1)
        # main ball
        cv2.circle(img, (int(ball["x"]), int(ball["y"])), radius, color, -1)
        # shine
        cv2.circle(img, (int(ball["x"]) - 4, int(ball["y"]) - 4), 3, (255, 255, 255), -1)

    # UI
    cv2.putText(img, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("AI Hand Bounce Game", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
