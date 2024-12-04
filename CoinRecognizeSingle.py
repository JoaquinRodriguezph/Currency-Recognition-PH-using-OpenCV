import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Button, LabelFrame
from tkinter import Tk, Label

# Initialize SIFT detector
sift = cv2.SIFT_create()

def load_template(name, path):
    print(f"Loading template for: {name} from {path}")
    template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        print(f"Warning: Template image {path} could not be loaded.")
        return name, None, None, None
    kp, desc = sift.detectAndCompute(template_image, None)
    return name, template_image, kp, desc

# Load and preprocess templates
def load_templates(template_files):
    print("Loading templates...")
    templates = {}
    with ThreadPoolExecutor(max_workers=4) as executor:  # Limit the number of threads
        futures = [executor.submit(load_template, name, path) for name, path in template_files.items()]
        for future in futures:
            name, template_image, kp, desc = future.result()
            if template_image is not None:
                templates[name] = (template_image, kp, desc)
    print("Finished loading templates.")
    return templates

def recognize_coin_with_homography(template_image, template_kp, template_desc, input_kp, input_desc, coin_name):
    print(f"Attempting to recognize: {coin_name}")
    
    # Match descriptors using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(template_desc, input_desc, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"Found {len(good_matches)} good matches for {coin_name}")

    # Require a minimum number of matches
    MIN_MATCH_COUNT = 20
    if len(good_matches) >= MIN_MATCH_COUNT:
        print(f"Enough matches found for {coin_name} ({len(good_matches)}/{MIN_MATCH_COUNT})")
        
        # Extract matched keypoints
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Define bounding box on the template
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        print(f"Recognized {coin_name}.")
        return True, coin_name, len(good_matches)
    else:
        print(f"Not enough matches for {coin_name} ({len(good_matches)}/{MIN_MATCH_COUNT})")
        return False, None, len(good_matches)

# Main function for detecting and recognizing coins
def detect_and_recognize_coins(image_path, template_files):
    print(f"Processing input image: {image_path}")
    
    # Load templates
    templates = load_templates(template_files)

    # Read and preprocess input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"Error: Input image {image_path} could not be loaded.")
        return
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect and compute keypoints/descriptors for the input image
    input_kp, input_desc = sift.detectAndCompute(gray, None)
    print(f"Computed {len(input_kp)} keypoints for the input image.")

    # Store matches for each template
    matches_dict = {}

    # Check against each template
    for coin_name, (template_image, template_kp, template_desc) in templates.items():
        success, recognized_coin, num_matches = recognize_coin_with_homography(
            template_image, template_kp, template_desc,
            input_kp, input_desc, coin_name
        )
        matches_dict[coin_name] = num_matches

    # Print the number of matches for each template
    print("Number of matches for each template:")
    for coin_name, num_matches in matches_dict.items():
        print(f"{coin_name}: {num_matches}")

    # Find the denomination with the highest number of matches
    best_match = max(matches_dict, key=matches_dict.get)
    print(f"Best match: {best_match} with {matches_dict[best_match]} matches")

    # Annotate the best match on the image
    cv2.putText(input_image, f"Best match: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print("Recognition complete. Displaying results...")
    cv2.imshow("Recognized Coins", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the relative paths for coin templates
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

# GUI Functions
def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
    )
    if file_path:
        try:
            detect_and_recognize_coins(file_path, template_files_coins)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create GUI
style = Style(theme='cosmo')
root = style.master
root.title("Coin Recognition")

# Set minimum and maximum size for the window
root.minsize(400, 200)

# Create LabelFrame
frame_top = LabelFrame(root, text="Image Operations", bootstyle=PRIMARY)
frame_top.pack(padx=10, pady=10, fill="both", expand=True)

# Create and place buttons and label using grid layout
button_load = Button(frame_top, text="Load Image", command=load_image, bootstyle=PRIMARY)
button_load.pack(padx=10, pady=10)

root.mainloop()