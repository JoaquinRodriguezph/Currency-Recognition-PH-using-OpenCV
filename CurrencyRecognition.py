#!/usr/bin/env python
import cv2
import numpy as np
import os


def recognize_bill_with_homography(template_kp, template_desc, input_kp, input_desc, bill_name, input_image):
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
    MIN_MATCH_COUNT = 30
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
        debug_matches = cv2.drawMatches(template_image, template_kp, input_image, input_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"debug_matches_{bill_name}.jpg", debug_matches)

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
    "1000 Peso Frontv1": "Templates/front/1000PesoFrontv1.png",
    "1000 Peso Frontv2": "Templates/front/1000PesoFrontv2.jpg",
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

# Load the input image
input_image_path = "Examples/20pesos.jpg"
input_image = cv2.imread(input_image_path)
gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Compute SIFT descriptors for the input image
input_kp, input_desc = sift.detectAndCompute(gray_input_image, None)

if input_desc is None or len(input_kp) == 0:
    print("No features detected in the input image.")
    exit()

# Match templates to the input image
matched = False
for bill_name, (template_image, template_kp, template_desc) in templates.items():
    is_match, annotated_image, label = recognize_bill_with_homography(
        template_kp, template_desc, input_kp, input_desc, bill_name, input_image
    )
    if is_match:
        matched = True
        print(f"Matched: {label}")
        cv2.putText(annotated_image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        break

if not matched:
    print("No matching template found.")
    annotated_image = input_image

# Save and display the result
output_image_path = "annotated_image.jpg"
cv2.imwrite(output_image_path, annotated_image)
print(f"Result saved to {output_image_path}")
cv2.imshow("Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
