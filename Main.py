import pyautogui
import PIL
from pynput import mouse
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

game_area = (1237, 421, 638, 394)
distance_area = (1473, 441, 160, 26)
area_left, area_top, area_width, area_height = game_area
distance_left, distance_top, distance_width, distance_height = distance_area

script_running = False
program_active = True
actual_distance = 0.0
target_distance = 100

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with {button}")

listener = mouse.Listener(on_click=on_click)
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

def grab_distance_image():
    with mss.mss() as sct:
        monitor = {"top": int(distance_top), "left": int(distance_left), "width": int(distance_width), "height": int(distance_height)}
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb).convert("L")
        return img

def check_distance():
    global actual_distance
    while program_active:
        if script_running:
            last_print_time = time.time()
            distance_image = grab_distance_image()
            try:
                distance_text = tesserocr.image_to_text(distance_image).strip()
                actual_distance = float(distance_text[:-7])
                print(f"Distance: {actual_distance}")
                elapsed_time = time.time() - last_print_time
                print(f"Time since last print: {elapsed_time:.2f} seconds\n")
            except Exception as e:
                print("Error parsing OCR:", e)
            time.sleep(0.1)
        else:
            time.sleep(0.2)

BODY_PARTS_TO_USE = {
    "Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4,
    "LShoulder":5, "LElbow":6, "LWrist":7, "MidHip":8, "RHip":9,
    "RKnee":10, "RAnkle":11, "LHip":12, "LKnee":13, "LAnkle":14
}

net = cv2.dnn.readNetFromCaffe(
    "models/pose/body_25/pose_deploy.prototxt",
    "models/pose/body_25/pose_iter_584000.caffemodel"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

thr = 0.1
frame_skip = 1

def track_pose():
    char_width = 420
    char_height = 300
    char_left = area_left + (area_width // 2) - (char_width // 2)
    char_top = area_top + (area_height // 2) - (char_height // 2) + 40
    game_region = {"top": char_top, "left": char_left, "width": char_width, "height": char_height}
    frame_count = 0
    previous_points = {}

    with mss.mss() as sct:
        while program_active:
            if script_running:
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                sct_img = sct.grab(game_region)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                in_width = 368
                in_height = 368
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
                        x = int(0.3*v[0] + 0.7*previous_points[k][0])
                        y = int(0.3*v[1] + 0.7*previous_points[k][1])
                        points[k] = (x, y)
                previous_points = points.copy()

                waist = points.get("MidHip")
                neck = points.get("Neck")

                def draw_leg(hip_point, knee_point, ankle_point, knee_length, ankle_length):
                    if hip_point is None or knee_point is None or ankle_point is None:
                        return
                    vec_hip_to_knee = np.array(knee_point) - np.array(hip_point)
                    vec_hip_to_knee = vec_hip_to_knee / np.linalg.norm(vec_hip_to_knee) * knee_length
                    fixed_knee = (int(hip_point[0] + vec_hip_to_knee[0]), int(hip_point[1] + vec_hip_to_knee[1]))

                    vec_knee_to_ankle = np.array(ankle_point) - np.array(knee_point)
                    vec_knee_to_ankle = vec_knee_to_ankle / np.linalg.norm(vec_knee_to_ankle) * ankle_length
                    fixed_ankle = (int(fixed_knee[0] + vec_knee_to_ankle[0]), int(fixed_knee[1] + vec_knee_to_ankle[1]))

                    cv2.line(frame, hip_point, fixed_knee, (0,255,255), 2)
                    cv2.line(frame, fixed_knee, fixed_ankle, (0,255,255), 2)

                if waist is not None and neck is not None:
                    cv2.line(frame, waist, neck, (0,255,255), 2)

                draw_leg(points.get("LHip"), points.get("LKnee"), points.get("LAnkle"), knee_length=65, ankle_length=65)
                draw_leg(points.get("RHip"), points.get("RKnee"), points.get("RAnkle"), knee_length=80, ankle_length=75)

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