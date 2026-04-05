import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ── Safety: pyautogui won't throw on edge hits
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove built-in delay for snappier response

# ── Config ─────────────────────────────────────────────────────────────────────
CAM_W, CAM_H       = 1280, 720
SCREEN_W, SCREEN_H = pyautogui.size()

# Active zone: inner 60% of frame to reduce edge jitter
MARGIN_X = int(CAM_W * 0.20)
MARGIN_Y = int(CAM_H * 0.20)
ZONE_W   = CAM_W - 2 * MARGIN_X
ZONE_H   = CAM_H - 2 * MARGIN_Y

SMOOTHING       = 7     # Higher = smoother but more lag (5–12 works well)
CLICK_THRESHOLD = 38    # Pixel distance between thumb & index tip to trigger click
SCROLL_SPEED    = 20    # Pixels per frame when scrolling
HOLD_FRAMES     = 6     # Frames pinch must hold before click fires (debounce)

# ── Hand Detector ───────────────────────────────────────────────────────────────
class HandDetector:
    def __init__(self):
        self.mp_hands  = mp.solutions.hands
        self.hands     = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        self.mp_draw   = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.results   = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return self

    def draw(self, frame):
        if self.results.multi_hand_landmarks:
            for lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return frame

    def get_landmarks(self, frame):
        """Returns dict {id: (x_px, y_px)} or empty dict."""
        lms = {}
        if self.results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for lm_id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                lms[lm_id] = (int(lm.x * w), int(lm.y * h))
        return lms

    def fingers_up(self, lms):
        """Returns list of booleans [thumb, index, middle, ring, pinky]."""
        if not lms:
            return [False] * 5
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        up = []
        # Thumb (x-axis)
        up.append(lms[4][0] < lms[3][0])
        # Rest (y-axis: tip above pip)
        for i in range(1, 5):
            up.append(lms[tips[i]][1] < lms[pips[i]][1])
        return up


# ── Utilities ───────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def map_to_screen(px, py):
    """Map camera active-zone coords → full screen coords."""
    sx = np.interp(px, [MARGIN_X, MARGIN_X + ZONE_W], [0, SCREEN_W])
    sy = np.interp(py, [MARGIN_Y, MARGIN_Y + ZONE_H], [0, SCREEN_H])
    return int(np.clip(sx, 0, SCREEN_W - 1)), int(np.clip(sy, 0, SCREEN_H - 1))

def draw_ui(frame, gesture, fps, click_progress, scroll_dir=None):
    h, w = frame.shape[:2]

    # Active zone rectangle
    cv2.rectangle(frame,
                  (MARGIN_X, MARGIN_Y),
                  (MARGIN_X + ZONE_W, MARGIN_Y + ZONE_H),
                  (80, 80, 80), 1)

    # FPS
    cv2.putText(frame, f"FPS {int(fps)}", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    # Gesture label
    color = {
        "MOVE":   (0, 220, 120),
        "CLICK":  (0, 120, 255),
        "SCROLL": (255, 180, 0),
        "IDLE":   (100, 100, 100),
    }.get(gesture, (255, 255, 255))

    cv2.putText(frame, gesture, (w - 150, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Click charge bar
    if click_progress > 0:
        bar_len = int((click_progress / HOLD_FRAMES) * 160)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_len, h - 30),
                      (0, 120, 255), cv2.FILLED)
        cv2.rectangle(frame, (20, h - 50), (180, h - 30), (80, 80, 80), 1)
        cv2.putText(frame, "HOLD TO CLICK", (20, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Scroll arrow
    if scroll_dir == "UP":
        cv2.putText(frame, "▲ SCROLL UP", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    elif scroll_dir == "DOWN":
        cv2.putText(frame, "▼ SCROLL DOWN", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

    return frame


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    detector = HandDetector()

    # Smoothing buffers
    prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2

    # State
    click_counter = 0
    clicked       = False
    prev_time     = time.time()

    print("Virtual Mouse running — press Q to quit")
    print(f"Screen: {SCREEN_W}x{SCREEN_H} | Active zone: {ZONE_W}x{ZONE_H}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        detector.process(frame)
        lms = detector.get_landmarks(frame)
        frame = detector.draw(frame)

        gesture    = "IDLE"
        scroll_dir = None

        if lms:
            fingers = detector.fingers_up(lms)
            ix, iy  = lms[8]   # Index tip
            tx, ty  = lms[4]   # Thumb tip
            mx, my  = lms[12]  # Middle tip

            # ── Gesture detection ──────────────────────────────────────────────

            # SCROLL: index + middle up, others down → vertical middle movement
            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                gesture    = "SCROLL"
                mid_y      = (iy + my) // 2
                scroll_dir = "UP" if mid_y < CAM_H // 2 else "DOWN"
                pyautogui.scroll(SCROLL_SPEED if scroll_dir == "UP" else -SCROLL_SPEED)

            # MOVE: only index up
            elif fingers[1] and not fingers[2]:
                gesture = "MOVE"

                # Map to screen
                sx, sy = map_to_screen(ix, iy)

                # Exponential smoothing
                smooth_x = prev_x + (sx - prev_x) / SMOOTHING
                smooth_y = prev_y + (sy - prev_y) / SMOOTHING
                prev_x, prev_y = smooth_x, smooth_y

                pyautogui.moveTo(int(smooth_x), int(smooth_y))

                # Visual: index tip circle
                cv2.circle(frame, (ix, iy), 10, (0, 220, 120), cv2.FILLED)

            # CLICK: pinch (thumb + index close together)
            if fingers[1] and not fingers[2]:
                pinch_d = dist(lms[4], lms[8])
                cv2.line(frame, lms[4], lms[8],
                         (0, 120, 255) if pinch_d < CLICK_THRESHOLD else (80, 80, 80), 2)

                if pinch_d < CLICK_THRESHOLD:
                    gesture       = "CLICK"
                    click_counter += 1
                    if click_counter == HOLD_FRAMES and not clicked:
                        pyautogui.click()
                        clicked = True
                else:
                    click_counter = max(0, click_counter - 2)  # Decay faster
                    clicked       = False

        # FPS
        now      = time.time()
        fps      = 1 / (now - prev_time + 1e-9)
        prev_time = now

        frame = draw_ui(frame, gesture, fps,
                        min(click_counter, HOLD_FRAMES), scroll_dir)

        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()