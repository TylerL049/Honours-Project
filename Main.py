import pyautogui
import PIL
from pynput import mouse
import time
from PIL import Image
import threading
from io import BytesIO
import mss
import os
os.environ['TESSDATA_PREFIX'] = r'C:\Users\tyler\AppData\Local\Programs\Tesseract-OCR\tessdata'
import tesserocr
from pynput.keyboard import Key, Listener
import sys
import cv2
import numpy as np

print("Code Started\n")

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
                print("K key pressed, starting script...")
                script_running = True
        elif key.char == 'l':
            print("L key pressed, stopping script...")
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
        img.save("test_image.png")
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
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13
}

net = cv2.dnn.readNetFromCaffe("models/pose/pose_deploy_linevec.prototxt",
                               "models/pose/pose_iter_440000.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

in_width = 128
in_height = 128
thr = 0.1
frame_skip = 2

def track_pose():
    char_width, char_height = 300, 350
    char_left = area_left + (area_width // 2) - (char_width // 2) - 20
    char_top = area_top + (area_height // 2) - (char_height // 2) + 20
    game_region = {"top": char_top, "left": char_left, "width": char_width, "height": char_height}
    frame_count = 0

    with mss.mss() as sct:
        while program_active:
            if script_running:
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                sct_img = sct.grab(game_region)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                frame_height, frame_width = frame.shape[:2]
                frame_resized = cv2.resize(frame, (in_width, in_height))

                inp = cv2.dnn.blobFromImage(frame_resized, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
                net.setInput(inp)
                out = net.forward()

                points = {}
                for part_name, idx in BODY_PARTS_TO_USE.items():
                    heatMap = out[0, idx, :, :]
                    _, conf, _, point = cv2.minMaxLoc(heatMap)
                    x = int(frame_width * point[0] / out.shape[3])
                    y = int(frame_height * point[1] / out.shape[2])
                    if conf > thr:
                        points[part_name] = (x, y)
                    else:
                        points[part_name] = None

                left_foot = points.get("LAnkle")
                right_foot = points.get("RAnkle")
                left_knee = points.get("LKnee")
                right_knee = points.get("RKnee")
                left_hip = points.get("LHip")
                right_hip = points.get("RHip")
                neck = points.get("Neck")

                if left_hip and right_hip:
                    waist = (int((left_hip[0] + right_hip[0]) / 2), int((left_hip[1] + right_hip[1]) / 2))
                else:
                    waist = None

                if left_foot and left_knee:
                    cv2.line(frame, left_foot, left_knee, (0, 255, 255), 2)
                if right_foot and right_knee:
                    cv2.line(frame, right_foot, right_knee, (0, 255, 255), 2)
                if waist:
                    if left_knee:
                        cv2.line(frame, left_knee, waist, (0, 255, 255), 2)
                    if right_knee:
                        cv2.line(frame, right_knee, waist, (0, 255, 255), 2)
                if waist and neck:
                    cv2.line(frame, waist, neck, (0, 255, 255), 2)

                cv2.imshow("QWOP Pose Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.2)

    cv2.destroyAllWindows()

print("Waiting to start (press 'K' to begin)...")

distance_thread = threading.Thread(target=check_distance, daemon=True)
pose_thread = threading.Thread(target=track_pose, daemon=True)
distance_thread.start()
pose_thread.start()

while program_active:
    time.sleep(1)

print("Script stopped.")