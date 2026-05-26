import argparse
import sys
import time

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
from pyglet.window import key


# game values
WINDOW_TITLE = 'AR Whack-a-Mole'
MOLE_RADIUS = 36
MOLE_LIFETIME = 1.4
MAX_MISSES = 5
HIT_PADDING = 12
SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)
SKIN_HSV_LOWER = np.array([0, 30, 60], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([25, 180, 255], dtype=np.uint8)
FINGER_MIN_AREA = 500
FINGER_MAX_AREA_RATIO = 0.35
FINGER_SMOOTH_ALPHA = 0.55
FINGER_HOLD_FRAMES = 4
TARGET_HIT_RADIUS_BONUS = 20


# global
cap = None
window = None
state = None
latest_texture = None


class GameState:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.target = None
        self.target_spawn_time = 0.0
        self.finger_point = None
        self.finger_missing_frames = 0
        self.rng = np.random.default_rng()
        self.spawn_target()

    def spawn_target(self, avoid_point=None):
        margin = MOLE_RADIUS + 30
        x_min = margin
        x_max = max(margin + 1, self.width - margin)
        y_min = margin
        y_max = max(margin + 1, self.height - margin)

        chosen = None
        for _ in range(20):
            x = int(self.rng.integers(x_min, x_max))
            y = int(self.rng.integers(y_min, y_max))
            chosen = np.array([x, y], dtype=np.float32)
            if avoid_point is None:
                break
            if np.linalg.norm(chosen - np.array(avoid_point, dtype=np.float32)) > MOLE_RADIUS * 4:
                break

        if chosen is None:
            chosen = np.array([self.width // 2, self.height // 2], dtype=np.float32)

        self.target = {
            'center': chosen,
            'radius': MOLE_RADIUS,
        }
        self.target_spawn_time = time.perf_counter()

    def restart(self):
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.finger_point = None
        self.finger_missing_frames = 0
        self.spawn_target()

    def update_finger_tracking(self, detected_point):
        # keeps a short history so brief detection dropouts donot jitter the pointer.

        if detected_point is None:
            self.finger_missing_frames += 1
            if self.finger_missing_frames > FINGER_HOLD_FRAMES:
                self.finger_point = None
            return self.finger_point

        self.finger_missing_frames = 0

        if self.finger_point is None:
            self.finger_point = detected_point
            return self.finger_point

        px, py = self.finger_point
        nx, ny = detected_point
        sx = int(round((1.0 - FINGER_SMOOTH_ALPHA) * px + FINGER_SMOOTH_ALPHA * nx))
        sy = int(round((1.0 - FINGER_SMOOTH_ALPHA) * py + FINGER_SMOOTH_ALPHA * ny))

        sx = max(0, min(self.width - 1, sx))
        sy = max(0, min(self.height - 1, sy))
        self.finger_point = (sx, sy)
        return self.finger_point

    def update(self, finger_point):
        now = time.perf_counter()
        tracked_finger = self.update_finger_tracking(finger_point)

        if self.game_over:
            return

        # checks for hit then time out target.
        if tracked_finger is not None:
            finger = np.array(tracked_finger, dtype=np.float32)
            target = np.array(self.target['center'], dtype=np.float32)
            hit_distance = self.target['radius'] + HIT_PADDING + TARGET_HIT_RADIUS_BONUS

            if np.linalg.norm(finger - target) <= hit_distance:
                self.score += 1
                self.spawn_target(avoid_point=tracked_finger)
                return

        if now - self.target_spawn_time > MOLE_LIFETIME:
            self.misses += 1
            if self.misses >= MAX_MISSES:
                self.game_over = True
                return
            self.spawn_target(avoid_point=tracked_finger)
            return


# parse video device
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-id', type=int, default=0)
    return parser.parse_args()


# converts an OpenCV BGR frame into a pyglet texture.
def cv2glet(img, fmt):
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols,
                                   height=rows,
                                   fmt=fmt,
                                   data=raw_img,
                                   pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg


def create_detector():
    # use same Aruco dictionary every frame
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()

    if hasattr(aruco, 'ArucoDetector'):
        return aruco.ArucoDetector(aruco_dict, aruco_params)
    return aruco_dict, aruco_params


def detect_markers(gray, detector):
    if hasattr(detector, 'detectMarkers'):
        return detector.detectMarkers(gray)

    aruco_dict, aruco_params = detector
    return aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)


def order_marker_indices(centers):
    # sort the four detected markers 
    sums = centers.sum(axis=1)
    diffs = np.diff(centers, axis=1).ravel()
    tl = int(np.argmin(sums))
    br = int(np.argmax(sums))
    tr = int(np.argmin(diffs))
    bl = int(np.argmax(diffs))
    return [tl, tr, br, bl]


def get_board_points(corners):
    if len(corners) < 4:
        return None

    # picks the inner corner of each marker to define the board quad.
    marker_contours = [c.reshape(4, 2).astype(np.float32) for c in corners]
    centers = np.array([marker.mean(axis=0) for marker in marker_contours], dtype=np.float32)
    order = order_marker_indices(centers)
    ordered_markers = [marker_contours[index] for index in order]
    board_center = np.mean(centers[order], axis=0)

    source_points = []
    for marker in ordered_markers:
        distances = np.linalg.norm(marker - board_center, axis=1)
        source_points.append(marker[int(np.argmin(distances))])

    return np.array(source_points, dtype=np.float32)


def warp_board(frame, corners):
    # warp board to the camera's resolution.
    board_points = get_board_points(corners)
    if board_points is None:
        return None

    height, width = frame.shape[:2]
    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(board_points, destination)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped


def detect_finger_point(frame):
    # builds a skin mask and keeps the most contour
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER)
    mask_hsv = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    height, width = frame.shape[:2]
    frame_area = float(height * width)
    best_contour = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < FINGER_MIN_AREA:
            continue
        if area > frame_area * FINGER_MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x <= 1 or y <= 1 or (x + w) >= (width - 1) or (y + h) >= (height - 1)

        score = area * (0.45 if touches_border else 1.0)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None, mask

    moments = cv2.moments(best_contour)
    if moments['m00'] == 0:
        return None, mask

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    hull_points = cv2.convexHull(best_contour, returnPoints=True).reshape(-1, 2)
    if len(hull_points) == 0:
        return None, mask

    center = np.array([cx, cy], dtype=np.float32)
    distances = np.linalg.norm(hull_points.astype(np.float32) - center, axis=1)
    fingertip = hull_points[int(np.argmax(distances))]
    finger_point = (int(fingertip[0]), int(fingertip[1]))
    return finger_point, mask


def draw_hud(frame):
    cv2.putText(frame, f'Score: {state.score}', (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f'Misses: {state.misses}/{MAX_MISSES}', (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def draw_game_over_overlay(frame):
    # draws end screen 
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    panel_w = min(520, width - 40)
    panel_h = 210
    panel_x = (width - panel_w) // 2
    panel_y = (height - panel_h) // 2

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 255), 3)

    cv2.putText(frame, 'Game Over', (panel_x + 28, panel_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(frame, f'Score: {state.score}', (panel_x + 28, panel_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f'Misses: {state.misses}/{MAX_MISSES}', (panel_x + 28, panel_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, 'Press R or SPACE to restart', (panel_x + 28, panel_y + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def draw_target(frame):
    # draws bullseye targets
    if state.target is None:
        return

    center = tuple(int(value) for value in state.target['center'])
    radius = int(state.target['radius'])

    cv2.circle(frame, center, radius + 10, (0, 0, 0), -1)
    cv2.circle(frame, center, radius, (0, 0, 255), -1)
    cv2.circle(frame, center, max(8, radius // 3), (0, 255, 255), -1)


def draw_finger(frame, finger_point):
    # highlights the tracked fingertip with a crosshair
    if finger_point is None:
        return

    cv2.circle(frame, finger_point, 10, (0, 255, 0), 2)
    cv2.line(frame, (finger_point[0] - 14, finger_point[1]), (finger_point[0] + 14, finger_point[1]), (0, 255, 0), 2)
    cv2.line(frame, (finger_point[0], finger_point[1] - 14), (finger_point[0], finger_point[1] + 14), (0, 255, 0), 2)


def make_placeholder(width, height, message):
    # fallback frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def draw_marker_setup_overlay(frame, marker_count):
    # shows raw image while user aligns all board markers in the camera 
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 96), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, f'Markers visible: {marker_count}/4', (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, 'Show all four ArUco corner markers to start the game', (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def process_frame(frame, detector):
    # detect markers + warp the board + pointer position.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detect_markers(gray, detector)

    marker_count = 0 if ids is None else len(ids)
    setup_view = frame.copy()
    if marker_count > 0:
        aruco.drawDetectedMarkers(setup_view, corners, ids)
    setup_view = draw_marker_setup_overlay(setup_view, marker_count)

    if ids is None or len(corners) < 4:
        return None, None, None, setup_view

    warped = warp_board(frame, corners)
    if warped is None:
        return None, None, None, setup_view

    finger_point, mask = detect_finger_point(warped)
    return warped, finger_point, mask, setup_view


def update(dt):
    global latest_texture

    ret, frame = cap.read()
    if not ret or frame is None:
        return

    warped, finger_point, _, setup_view = process_frame(frame, detector)

    # keeps game-over screen visible
    if state.game_over:
        output = warped.copy() if warped is not None else setup_view
        output = draw_game_over_overlay(output)
    elif warped is None:
        output = setup_view
    else:
        state.last_warp = warped.copy()
        state.update(finger_point)
        output = warped.copy()
        draw_target(output)
        draw_finger(output, state.finger_point)
        draw_hud(output)

    latest_texture = cv2glet(output, 'BGR')


def on_draw():
    window.clear()
    if latest_texture is not None:
        latest_texture.blit(0, 0, 0)


def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        pyglet.app.exit()
    elif symbol == key.SPACE:
        state.restart()


def on_close():
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    pyglet.app.exit()


def main():
    global cap, detector, state, window, latest_texture

    args = get_args()
    detector = create_detector()

    cap = cv2.VideoCapture(args.video_id)
    if not cap.isOpened():
        print(f'Failed to open webcam {args.video_id}')
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print('Failed to read from webcam')
        cap.release()
        sys.exit(1)

    height, width = frame.shape[:2]
    state = GameState(width, height)

    window = pyglet.window.Window(width=width, height=height, caption=WINDOW_TITLE, resizable=False)
    window.push_handlers(on_draw=on_draw, on_key_press=on_key_press, on_close=on_close)

    latest_texture = cv2glet(make_placeholder(width, height, 'Starting camera...'), 'BGR')

    pyglet.clock.schedule_interval(update, 1.0 / 30.0)
    pyglet.app.run()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print('Interrupted')
