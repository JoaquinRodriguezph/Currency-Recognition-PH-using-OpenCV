#!/usr/bin/env python
import cv2
import numpy as np
import os
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Button, LabelFrame
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
import zipfile
import shutil


# Global variables to store input image and its features
input_image = None
gray_input_image = None
input_kp = None
input_desc = None

def recognize_bill_with_homography(template_image, template_kp, template_desc, input_kp, input_desc, bill_name, input_image):
    if template_desc is None or input_desc is None:
        print(f"Descriptors are None for {bill_name}")
        return False, input_image, None, 0, 0.0

    if len(template_desc) == 0 or len(input_desc) == 0:
        print(f"No descriptors found for {bill_name}")
        return False, input_image, None, 0, 0.0

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(template_desc, input_desc, k=2)
    except Exception as e:
        print(f"Error during knnMatch for {bill_name}: {e}")
        return False, input_image, None, 0, 0.0

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_count = len(good_matches)
    inlier_ratio = match_count / len(matches) if matches else 0.0

    MIN_MATCH_COUNT = 20
    if match_count >= MIN_MATCH_COUNT:
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        input_image = cv2.polylines(input_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        return True, input_image, bill_name, match_count, inlier_ratio
    else:
        return False, input_image, None, match_count, inlier_ratio

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

templates = {}
for name, path in template_files.items():
    template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        print(f"Error loading image {path}")
        continue
    template_kp, template_desc = sift.detectAndCompute(template_image, None)
    templates[name] = (template_image, template_kp, template_desc)


# GUI Functions
def load_image():
    global input_image, gray_input_image, input_kp, input_desc
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    if file_path:
        try:
            input_image = cv2.imread(file_path)
            if input_image is None:
                messagebox.showerror("Error", "Could not load the image. Please try another file.")
                return
            
            gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            input_kp, input_desc = sift.detectAndCompute(gray_input_image, None)
            
            if input_kp is None or len(input_kp) == 0:
                messagebox.showwarning("Warning", "No features detected in the image.")
                return
            
            display_image(input_image)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def process_image():
    global input_image
    
    if input_image is None or input_desc is None or len(input_kp) == 0:
        messagebox.showwarning("Warning", "Please load an image with detectable features.")
        return

    annotated_image = input_image.copy()
    match_scores = {}

    for bill_name, (template_image, template_kp, template_desc) in templates.items():
        is_match, temp_image, label, match_count, inlier_ratio = recognize_bill_with_homography(
            template_image, template_kp, template_desc, input_kp, input_desc, bill_name, annotated_image
        )
        if is_match:
            normalized_count = match_count / len(template_kp)
            score = (0.7 * normalized_count) + (0.3 * inlier_ratio)
            match_scores[label] = score

    if match_scores:
        best_match = max(match_scores, key=match_scores.get)
        print(f"Best Match: {best_match} with score: {match_scores[best_match]:.2f}")
        cv2.putText(annotated_image, f"Best Match: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        display_image(annotated_image)
    else:
        messagebox.showinfo("Result", "No matching template found.")
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
            is_match, annotated_image, label, match_count, inlier_ratio = recognize_bill_with_homography(
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


def process_zip_file():
    global input_image, gray_input_image, input_kp, input_desc
    
    # Prompt user to select a ZIP file
    zip_file_path = filedialog.askopenfilename(
        filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
    )
    if not zip_file_path:
        return

    # Create a temporary directory to extract and process images
    temp_dir = "temp_images"
    processed_dir = "processed_images"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    try:
        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process each image in the extracted directory
        for root_dir, _, files in os.walk(temp_dir):
            for file_name in files:
                file_path = os.path.join(root_dir, file_name)
                try:
                    input_image = cv2.imread(file_path)
                    if input_image is None:
                        print(f"Skipping non-image file: {file_name}")
                        continue
                    
                    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                    input_kp, input_desc = sift.detectAndCompute(gray_input_image, None)

                    annotated_image = input_image.copy()
                    match_scores = {}

                    # Match with templates
                    for bill_name, (template_image, template_kp, template_desc) in templates.items():
                        is_match, temp_image, label, match_count, inlier_ratio = recognize_bill_with_homography(
                            template_image, template_kp, template_desc, input_kp, input_desc, bill_name, annotated_image
                        )
                        if is_match:
                            normalized_count = match_count / len(template_kp)
                            score = (0.7 * normalized_count) + (0.3 * inlier_ratio)
                            match_scores[label] = score
                    
                    # Annotate the best match if found
                    if match_scores:
                        best_match = max(match_scores, key=match_scores.get)
                        print(f"Best Match for {file_name}: {best_match}")
                        cv2.putText(annotated_image, f"Best Match: {best_match}", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Save the processed image
                    processed_path = os.path.join(processed_dir, file_name)
                    cv2.imwrite(processed_path, annotated_image)

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        
        # Create a ZIP file for the processed images
        output_zip = "processed_images.zip"
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, _, files in os.walk(processed_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, processed_dir)
                    zipf.write(file_path, arcname)

        messagebox.showinfo("Success", f"Processed images saved in '{output_zip}'")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary directories
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)

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

button_camera = Button(frame_bottom, text="Activate Bills Camera", command=activate_camera, bootstyle=INFO)
button_camera.grid(row=0, column=0, padx=10, pady=10)

button_process_zip = Button(frame_top, text="Process ZIP File", command=process_zip_file, bootstyle=WARNING)
button_process_zip.grid(row=0, column=2, padx=10, pady=10)

root.mainloop()