import cv2
import numpy as np
from utils import calculate_lengths_and_area



def process_plot(image_path, tap_point, scale_pixels, scale_feet):
    print("🔍 Tap Point:", tap_point)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("❌ Failed to load image. Check image path.")

    print("📏 Original Image Shape:", image.shape)
    print("📍 Tap Point Received (should be scaled):", tap_point)

    # 🔧 Step 1: Improved Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional Gaussian blur to smooth image before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to isolate dark lines
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Slight dilation to enhance boundaries
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # 🧱 Step 2: Contour Detection with Hierarchy
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print("🟢 Total contours found:", len(contours))

    # 🔀 Split contours into OUTER and INNER based on hierarchy
    outer_contours = []
    inner_contours = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        parent_idx = hierarchy[0][idx][3]

        if area < 100:
            continue

        if parent_idx == -1:
            outer_contours.append(cnt)
            print(f"🔵 OUTER Contour #{idx}, Area={area:.1f}")
        else:
            inner_contours.append(cnt)
            print(f"🟡 INNER Contour #{idx}, Area={area:.1f}")

    print(f"✅ Outer Contours: {len(outer_contours)}, Inner Contours: {len(inner_contours)}")

    # 📸 Save debug image
    debug_img = image.copy()
    cv2.drawContours(debug_img, outer_contours, -1, (255, 0, 0), 2)  # 🔵 Blue = outer
    cv2.drawContours(debug_img, inner_contours, -1, (0, 0, 255), 2)  # 🔴 Red = inner
    cv2.imwrite("debug_inner_outer.jpg", debug_img)

    # 🔍 Step 3: Tap Point Inside Check (Prioritize inner-most)
    selected_contour = None
    for idx, contour in enumerate(inner_contours + outer_contours):
        inside = cv2.pointPolygonTest(contour, tap_point, False)
        if inside > 0:
            area = cv2.contourArea(contour)
            if selected_contour is None or area < cv2.contourArea(selected_contour):
                selected_contour = contour
                print(f"✅ Tap matched with contour #{idx}, Area={area:.1f}")

    # 🧭 Step 4: Distance fallback
    if selected_contour is None:
        print("⚠️ Tap not inside any contour. Checking for nearby contours...")
        min_distance = float("inf")
        for cnt in inner_contours + outer_contours:
            dist = cv2.pointPolygonTest(cnt, tap_point, True)
            if abs(dist) < 10:
                if abs(dist) < min_distance:
                    min_distance = abs(dist)
                    selected_contour = cnt
        if selected_contour is not None:
            print(f"✅ Selected nearest contour within {min_distance:.1f} px of tap.")
        else:
            raise ValueError("❌ No plot boundary found near the tap point.")

    # 🔷 Step 5: Approximate and Extract Points
    epsilon = 0.0001 * cv2.arcLength(selected_contour, True)
    approx = cv2.approxPolyDP(selected_contour, epsilon, True)
    points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    print("📍 Final Contour Points:", points)

    # 📐 Step 6: Area + Length Calculation
    data = calculate_lengths_and_area(points, scale_pixels, scale_feet)
    data["boundary_pixels"] = points  # 🟩 Add this line to include polygon points in response

    # 🖼 Optional: Draw and save the final smooth contour
    smooth_debug = image.copy()
    cv2.drawContours(smooth_debug, [approx], -1, (0, 255, 0), 2)  # Green
    cv2.imwrite("debug_smooth_contour.jpg", smooth_debug)

    return data
