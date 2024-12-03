import cv2
import numpy as np
import os

# Initialize SIFT detector
sift = cv2.SIFT_create()

def recognize_coin_with_homography(template_image, template_kp, template_desc, input_kp, input_desc, coin_name, input_image):
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

        # Draw bounding box and annotate
        input_image = cv2.polylines(input_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(input_image, coin_name, tuple(np.int32(dst[0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"Recognized {coin_name} and drew bounding box.")
        return True, input_image, coin_name
    else:
        print(f"Not enough matches for {coin_name} ({len(good_matches)}/{MIN_MATCH_COUNT})")
        return False, input_image, None

# Load and preprocess templates
def load_templates(template_files):
    print("Loading templates...")
    templates = {}
    for name, path in template_files.items():
        print(f"Loading template for: {name} from {path}")
        template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template_image is None:
            print(f"Warning: Template image {path} could not be loaded.")
            continue
        kp, desc = sift.detectAndCompute(template_image, None)
        templates[name] = (template_image, kp, desc)
    print("Finished loading templates.")
    return templates

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
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 150)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)

    # Detect contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} contours in the input image.")

    # Detect and compute keypoints/descriptors for the input image
    input_kp, input_desc = sift.detectAndCompute(gray, None)
    print(f"Computed {len(input_kp)} keypoints for the input image.")

    # Loop through contours and attempt to recognize coins
    for i, contour in enumerate(contours):
        print(f"Processing contour {i + 1}/{len(contours)}")
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]

        # Check against each template
        for coin_name, (template_image, template_kp, template_desc) in templates.items():
            success, input_image, recognized_coin = recognize_coin_with_homography(
                template_image, template_kp, template_desc,
                input_kp, input_desc, coin_name, input_image
            )
            if success:
                print(f"Coin recognized: {recognized_coin}")
                break  # If a match is found, stop checking further templates

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

# Run the recognition system
image_path = "coin.jpg"  # Replace with your test image
detect_and_recognize_coins(image_path, template_files_coins)
