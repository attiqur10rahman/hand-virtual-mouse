import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import urllib.request
import os

MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Model downloading... please wait...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done!")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

CAM_W, CAM_H = 1280, 720
SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_X = int(CAM_W * 0.20)
MARGIN_Y = int(CAM_H * 0.20)
ZONE_W = CAM_W - 2 * MARGIN_X
ZONE_H = CAM_H - 2 * MARGIN_Y
SMOOTHING = 7
CLICK_THRESHOLD = 38
SCROLL_SPEED = 20
HOLD_FRAMES = 6

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

def dist(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

def map_to_screen(px, py):
    sx = np.interp(px, [MARGIN_X, MARGIN_X+ZONE_W], [0, SCREEN_W])
    sy = np.interp(py, [MARGIN_Y, MARGIN_Y+ZONE_H], [0, SCREEN_H])
    return int(np.clip(sx,0,SCREEN_W-1)), int(np.clip(sy,0,SCREEN_H-1))

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    lms = {}
    if result.hand_landmarks:
        h, w = frame.shape[:2]
        for i, lm in enumerate(result.hand_landmarks[0]):
            lms[i] = (int(lm.x*w), int(lm.y*h))
    return lms

def fingers_up(lms):
    if not lms: return [False]*5
    tips = [4,8,12,16,20]
    pips = [3,6,10,14,18]
    up = [lms[4][0] < lms[3][0]]
    for i in range(1,5):
        up.append(lms[tips[i]][1] < lms[pips[i]][1])
    return up

def draw_hand(frame, lms):
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
             (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    for a,b in conns:
        if a in lms and b in lms:
            cv2.line(frame, lms[a], lms[b], (80,80,80), 1)
    for x,y in lms.values():
        cv2.circle(frame, (x,y), 4, (200,200,200), cv2.FILLED)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    prev_x, prev_y = SCREEN_W//2, SCREEN_H//2
    click_counter = 0
    clicked = False
    prev_time = time.time()
    print("Virtual Mouse running! Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        lms = get_landmarks(frame)
        frame = draw_hand(frame, lms)
        gesture = "IDLE"

        if lms:
            fingers = fingers_up(lms)
            ix, iy = lms[8]
            mx, my = lms[12]

            if fingers[1] and fingers[2] and not fingers[3]:
                gesture = "SCROLL"
                mid_y = (iy+my)//2
                pyautogui.scroll(SCROLL_SPEED if mid_y < CAM_H//2 else -SCROLL_SPEED)

            elif fingers[1] and not fingers[2]:
                gesture = "MOVE"
                sx, sy = map_to_screen(ix, iy)
                prev_x += (sx-prev_x)/SMOOTHING
                prev_y += (sy-prev_y)/SMOOTHING
                pyautogui.moveTo(int(prev_x), int(prev_y))
                cv2.circle(frame, (ix,iy), 10, (0,220,120), cv2.FILLED)
                pd = dist(lms[4], lms[8])
                cv2.line(frame, lms[4], lms[8], (0,120,255) if pd<CLICK_THRESHOLD else (80,80,80), 2)
                if pd < CLICK_THRESHOLD:
                    gesture = "CLICK"
                    click_counter += 1
                    if click_counter == HOLD_FRAMES and not clicked:
                        pyautogui.click()
                        clicked = True
                else:
                    click_counter = max(0, click_counter-2)
                    clicked = False

        fps = 1/(time.time()-prev_time+1e-9)
        prev_time = time.time()
        color = {"MOVE":(0,220,120),"CLICK":(0,120,255),"SCROLL":(255,180,0),"IDLE":(100,100,100)}.get(gesture)
        cv2.putText(frame, f"FPS {int(fps)}", (20,36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
        cv2.putText(frame, gesture, (CAM_W-150,36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()