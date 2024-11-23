#!/usr/bin/env python
import cv2
import numpy as np
import os
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Button, LabelFrame
from tkinter import filedialog, Label
from PIL import Image, ImageTk

def recognize_bill_with_homography(template_image, template_kp, template_desc, input_kp, input_desc, bill_name, input_image):
    # Match descriptors using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of times the tree is recursively searched
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(template_desc, input_desc, k=2)

    # Apply ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Require a minimum number of matches to proceed
    MIN_MATCH_COUNT = 20    # temporarily lowered the count to 20 so the video feed
                            # can detect the bill. was originally 30
    if len(good_matches) >= MIN_MATCH_COUNT:
        # Extract the matched keypoints
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Draw the bounding box for the matched area
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        # debug_matches = cv2.drawMatches(template_image, template_kp, input_image, input_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite(f"debug_matches_{bill_name}.jpg", debug_matches)
        input_image = cv2.polylines(input_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        return True, input_image, bill_name
    else:
        print(f"Not enough matches found for {bill_name} - {len(good_matches)} / {MIN_MATCH_COUNT}")
        return False, input_image, None

# Set the working directory to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize SIFT
sift = cv2.SIFT_create()

# Load template images and compute descriptors
print("Opening template images...")
template_files = {
    "1000 Peso Front": "Templates/front/1000PesoFrontv2.jpg",
    "500 Peso Front": "Templates/front/500PesoFront.jpg",
    "200 Peso Front": "Templates/front/200PesoFront.png",
    "100 Peso Front": "Templates/front/100PesoFront.jpg",
    "50 Peso Front": "Templates/front/50PesoFront.jpg",
    "20 Peso Front": "Templates/front/20PesoFront.jpg",
    "1000 Peso Back": "Templates/back/1000PesoBack.jpg",
    "500 Peso Back": "Templates/back/500PesoBack.jpg",
    "200 Peso Back": "Templates/back/200PesoBack.jpg",
    "100 Peso Back": "Templates/back/100PesoBack.jpg",
    "50 Peso Back": "Templates/back/50PesoBack.jpg",
    "20 Peso Back": "Templates/back/20PesoBack.jpg"
}

template_files_coins = {
    "20 Peso Coin Frontv1": "Templates/front/20PesoCoinFront.png",
    "20 Peso Coin Frontv2": "Templates/front/20PesoCoinFrontv2.png",
    "10 Peso Coin Frontv1": "Templates/front/10PesoCoinFront.png",
    "10 Peso Coin Frontv2": "Templates/front/10PesoCoinFrontv2.png",
    "5 Peso Coin Frontv1": "Templates/front/5PesoCoinFront.png",
    "5 Peso Coin Frontv2": "Templates/front/5PesoCoinFrontv2.png",
    "1 Peso Coin Frontv1": "Templates/front/1PesoCoinFront.png",
    "1 Peso Coin Frontv2": "Templates/front/1PesoCoinFrontv2.png",
    "20 Peso Coin Backv1": "Templates/back/20PesoCoinBack.png",
    "20 Peso Coin Backv2": "Templates/back/20PesoCoinBackv2.png",
    "10 Peso Coin Backv1": "Templates/back/10PesoCoinBack.png",
    "10 Peso Coin Backv2": "Templates/back/10PesoCoinBackv2.png",
    "5 Peso Coin Backv1": "Templates/back/5PesoCoinBack.png",
    "5 Peso Coin Backv2": "Templates/back/5PesoCoinBackv2.png",
    "1 Peso Coin Backv1": "Templates/back/1PesoCoinBack.png",
    "1 Peso Coin Backv2": "Templates/back/1PesoCoinBackv2.png"
}

templates = {}
for name, path in template_files.items():
    template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        print(f"Error loading image {path}")
        continue
    template_kp, template_desc = sift.detectAndCompute(template_image, None)
    templates[name] = (template_image, template_kp, template_desc)

# Function to recognize coins using homography
def recognize_coin_with_homography(template_image, template_kp, template_desc, input_kp, input_desc, coin_name, input_image):
    # Match descriptors using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of times the tree is recursively searched
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(template_desc, input_desc, k=2)

    # Apply ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Require a minimum number of matches to proceed
    MIN_MATCH_COUNT = 15  # Adjusted for coins
    if len(good_matches) >= MIN_MATCH_COUNT:
        # Extract the matched keypoints
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Draw the bounding box for the matched area
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        input_image = cv2.polylines(input_image, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        return True, input_image, coin_name
    else:
        print(f"Not enough matches found for {coin_name} - {len(good_matches)} / {MIN_MATCH_COUNT}")
        return False, input_image, None

# Load coin templates
print("Opening coin template images...")
coin_templates = {}
for name, path in template_files_coins.items():
    template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        print(f"Error loading coin image {path}")
        continue
    template_kp, template_desc = sift.detectAndCompute(template_image, None)
    coin_templates[name] = (template_image, template_kp, template_desc)

# GUI Functions
def load_image():
    global input_image, gray_input_image, input_kp, input_desc
    file_path = filedialog.askopenfilename()
    if file_path:
        input_image = cv2.imread(file_path)
        gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_kp, input_desc = sift.detectAndCompute(gray_input_image, None)
        display_image(input_image)

# Update process_image to include coin recognition
def process_image():
    if input_desc is None or len(input_kp) == 0:
        print("No features detected in the input image.")
        return

    matched = False
    for bill_name, (template_image, template_kp, template_desc) in templates.items():
        is_match, annotated_image, label = recognize_bill_with_homography(
            template_image, template_kp, template_desc, input_kp, input_desc, bill_name, input_image
        )
        if is_match:
            matched = True
            print(f"Matched: {label}")
            cv2.putText(annotated_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            display_image(annotated_image)
            return

    for coin_name, (template_image, template_kp, template_desc) in coin_templates.items():
        is_match, annotated_image, label = recognize_coin_with_homography(
            template_image, template_kp, template_desc, input_kp, input_desc, coin_name, input_image
        )
        if is_match:
            matched = True
            print(f"Matched: {label}")
            cv2.putText(annotated_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            display_image(annotated_image)
            return

    if not matched:
        print("No matching template found.")
        display_image(input_image)


def display_image(image):
    fixed_width = 600
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(fixed_width * aspect_ratio)
    resized_image = cv2.resize(image, (fixed_width, new_height))

    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    label_image.config(image=image_tk)
    label_image.image = image_tk

def activate_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_kp, input_desc = sift.detectAndCompute(gray_frame, None)

        if input_desc is None or len(input_kp) == 0:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        matched = False
        for bill_name, (template_image, template_kp, template_desc) in templates.items():
            is_match, annotated_image, label = recognize_bill_with_homography(
                template_image, template_kp, template_desc, input_kp, input_desc, bill_name, frame
            )
            if is_match:
                matched = True
                cv2.putText(annotated_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Camera", annotated_image)
                break

        if not matched:
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


    cap.release()
    cv2.destroyAllWindows()

# Create GUI
style = Style(theme='cosmo') 
root = style.master
root.title("Currency Recognition")

# Set minimum and maximum size for the window
root.minsize(800, 600)

# Create LabelFrames
frame_top = LabelFrame(root, text="Image Operations", bootstyle=PRIMARY)
frame_top.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

frame_bottom = LabelFrame(root, text="Camera Operations", bootstyle=INFO)
frame_bottom.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Create and place buttons and label using grid layout
button_load = Button(frame_top, text="Load Image", command=load_image, bootstyle=PRIMARY)
button_load.grid(row=0, column=0, padx=10, pady=10)

button_process = Button(frame_top, text="Process Image", command=process_image, bootstyle=SUCCESS)
button_process.grid(row=0, column=1, padx=10, pady=10)

label_image = Label(frame_top)
label_image.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

button_camera = Button(frame_bottom, text="Activate Camera", command=activate_camera, bootstyle=INFO)
button_camera.grid(row=0, column=0, padx=10, pady=10)

root.mainloop()