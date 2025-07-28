import math

def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_lengths_and_area(points, scale_pixels, scale_feet):
    real_lengths = []
    total_pixel_length = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1)%len(points)]
        dist = pixel_distance(p1, p2)
        total_pixel_length += dist
        feet = (dist / scale_pixels) * scale_feet
        real_lengths.append(round(feet, 2))

    # Area using Shoelace formula (in pixels²)
    area_pixels = 0.5 * abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))))
    area_feet = (area_pixels / (scale_pixels**2)) * (scale_feet**2)

    return {
        "points": [(int(p[0]), int(p[1])) for p in points],  # NumPy int32 → Python int
        "side_lengths_ft": [float(length) for length in real_lengths],
        "perimeter_ft": float(round(sum(real_lengths), 2)),
        "area_sqft": float(round(area_feet, 2))
    }
