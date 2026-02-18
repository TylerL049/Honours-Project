import pyautogui
import PIL
from pynput import mouse, keyboard
import time
from PIL import Image
import threading
import mss
import os
os.environ['TESSDATA_PREFIX'] = r'C:\Users\tyler\AppData\Local\Programs\Tesseract-OCR\tessdata'
import tesserocr
from pynput.keyboard import Key, Listener
import cv2
import numpy as np
import re

game_area = (642, 190, 637, 396)
distance_area = (860, 206, 193, 29)
area_left, area_top, area_width, area_height = game_area
distance_left, distance_top, distance_width, distance_height = distance_area

script_running = False
program_active = True
actual_distance = 0.0
target_distance = 100

listener = mouse.Listener(on_click=lambda x,y,button,pressed: print(f"Mouse clicked at ({x}, {y}) with {button}") if pressed else None)
listener.start()

def on_press(key):
    global script_running, program_active
    try:
        if key.char == 'k':
            if not script_running:
                script_running = True
        elif key.char == 'l':
            script_running = False
            program_active = False
    except AttributeError:
        pass

def on_release(key):
    if key == Key.esc:
        return False

keyboard_listener = Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()

tess_api = tesserocr.PyTessBaseAPI()
try:
    tess_api.SetVariable("tessedit_char_whitelist", "0123456789.")
except Exception:
    pass
try:
    tess_api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
except Exception:
    pass

def grab_distance_image():
    with mss.mss() as sct:
        monitor = {"top": int(distance_top), "left": int(distance_left), "width": int(distance_width), "height": int(distance_height)}
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb).convert("L")
        return img

def check_distance():
    global actual_distance
    pattern = re.compile(r"[0-9]+(?:\.[0-9]+)?")
    while program_active:
        if script_running:
            start = time.time()
            distance_image = grab_distance_image()
            try:
                tess_api.SetImage(distance_image)
                distance_text = (tess_api.GetUTF8Text() or "").strip()
                tess_api.Clear()
                m = pattern.search(distance_text)
                if m:
                    actual_distance = float(m.group(0))
                    print(f"Distance: {actual_distance}")
                else:
                    print("Distance: parse failed")
                elapsed_time = time.time() - start
                print(f"Time since last print: {elapsed_time:.2f} seconds\n")
            except Exception as e:
                print("Error parsing OCR:", e)
            time.sleep(0.12)
        else:
            time.sleep(0.2)

BODY_PARTS_TO_USE = {
    "Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4,
    "LShoulder":5, "LElbow":6, "LWrist":7, "MidHip":8, "RHip":9,
    "RKnee":10, "RAnkle":11, "LHip":12, "LKnee":13, "LAnkle":14
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "models/pose/body_25/pose_deploy.prototxt"),
    os.path.join(BASE_DIR, "models/pose/body_25/pose_iter_584000.caffemodel")
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

thr = 0.1
frame_skip = 2
check_color = False

def color_on_press(key):
    global check_color
    try:
        if key.char == 'c':
            check_color = True
    except AttributeError:
        pass

def color_loop():
    global check_color
    while True:
        if check_color:
            x, y = pyautogui.position()
            pixel_color = pyautogui.screenshot().getpixel((x, y))
            print(f"Mouse at ({x},{y}) Color: {pixel_color}")
            check_color = False
        time.sleep(0.05)

color_keyboard_listener = keyboard.Listener(on_press=color_on_press)
color_keyboard_listener.start()
color_thread = threading.Thread(target=color_loop, daemon=True)
color_thread.start()

player_colors_rgb = [
    [86,51,25],[10,3,1],[146,83,61],[143,13,32],[106,130,149],
    [114,68,33],[60,16,11],[54,21,13],[74,23,16],[242,224,201],
    [146,154,173],[167,12,37],[154,13,37],[133,24,41],[171,35,58],
    [115,22,14],[26,10,2],[101,53,50],[67,16,12],[186,129,93],
    [91,32,20],[56,12,8],[76,26,15],[159,87,46],[181,11,47],
    [83,28,16],[195,209,228],[246,229,204],[203,225,249],[204,210,219]
]

remove_colors_rgb = [
    [204,83,40],[200,95,67],[198,113,87]
]

player_colors_hsv = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0][0] for c in player_colors_rgb]
remove_colors_hsv = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0][0] for c in remove_colors_rgb]

def remove_background_with_exclude(frame, player_colors_hsv, remove_colors_hsv, tol_h=4, tol_s=40, tol_v=40, remove_tol=2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color_hsv in player_colors_hsv:
        lower = np.array([max(color_hsv[0]-tol_h,0), max(color_hsv[1]-tol_s,0), max(color_hsv[2]-tol_v,0)], dtype=np.uint8)
        upper = np.array([min(color_hsv[0]+tol_h,179), min(color_hsv[1]+tol_s,255), min(color_hsv[2]+tol_v,255)], dtype=np.uint8)
        temp_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, temp_mask)
    for color_hsv in remove_colors_hsv:
        lower = np.array([max(color_hsv[0]-remove_tol,0), max(color_hsv[1]-remove_tol,0), max(color_hsv[2]-remove_tol,0)], dtype=np.uint8)
        upper = np.array([min(color_hsv[0]+remove_tol,179), min(color_hsv[1]+remove_tol,255), min(color_hsv[2]+remove_tol,255)], dtype=np.uint8)
        remove_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(remove_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def track_pose():
    char_width = 420
    char_height = 300
    char_left = area_left + (area_width // 2) - (char_width // 2)
    char_top = area_top + (area_height // 2) - (char_height // 2) + 40
    game_region = {"top": char_top, "left": char_left, "width": char_width, "height": char_height}
    frame_count = 0
    previous_points = {}
    ema_alpha = 0.4

    def draw_leg_safe(hip, knee, ankle):
        if hip is not None and knee is not None:
            cv2.line(frame, hip, knee, (0,255,255), 2)
        if knee is not None and ankle is not None:
            cv2.line(frame, knee, ankle, (0,255,255), 2)

    with mss.mss() as sct:
        while program_active:
            if script_running:
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                sct_img = sct.grab(game_region)
                arr = np.frombuffer(sct_img.rgb, dtype=np.uint8).reshape(sct_img.height, sct_img.width, 3)
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                frame = remove_background_with_exclude(frame, player_colors_hsv, remove_colors_hsv)
                in_width = 200
                in_height = 200
                frame_resized = cv2.resize(frame, (in_width, in_height))
                inp = cv2.dnn.blobFromImage(frame_resized, 1.0/255, (in_width, in_height), (0,0,0), swapRB=True, crop=False)
                net.setInput(inp)
                out = net.forward()
                points = {}
                hm_w, hm_h = out.shape[3], out.shape[2]
                scale_x = frame.shape[1] / in_width
                scale_y = frame.shape[0] / in_height
                for part_name, idx in BODY_PARTS_TO_USE.items():
                    heatMap = out[0, idx, :, :]
                    _, conf, _, point = cv2.minMaxLoc(heatMap)
                    px = point[0] / hm_w * in_width
                    py = point[1] / hm_h * in_height
                    x = int(px * scale_x)
                    y = int(py * scale_y)
                    if conf > thr:
                        points[part_name] = (x, y)
                    else:
                        points[part_name] = None
                for k, v in points.items():
                    if v is not None and k in previous_points and previous_points[k] is not None:
                        x = int(ema_alpha * v[0] + (1 - ema_alpha) * previous_points[k][0])
                        y = int(ema_alpha * v[1] + (1 - ema_alpha) * previous_points[k][1])
                        points[k] = (x, y)
                previous_points = points.copy()
                waist = points.get("MidHip")
                neck = points.get("Neck")
                if waist is not None and neck is not None:
                    cv2.line(frame, waist, neck, (0,255,255), 2)
                draw_leg_safe(points.get("LHip"), points.get("LKnee"), points.get("LAnkle"))
                draw_leg_safe(points.get("RHip"), points.get("RKnee"), points.get("RAnkle"))
                cv2.imshow("QWOP Pose Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.2)
    cv2.destroyAllWindows()

distance_thread = threading.Thread(target=check_distance, daemon=True)
pose_thread = threading.Thread(target=track_pose, daemon=True)
distance_thread.start()
pose_thread.start()

while program_active:
    time.sleep(1)