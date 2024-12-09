import cv2 as cv
import numpy as np
import pyautogui

# Load the templates
dino_template = cv.imread('dino.jpg', 0)
all_cartoons = ['cactus_small.png', 'cactus_medium.png', 'double_cactus.png', 'trio_cactus.png']
region = (0, 0, 1920, 1080)  # Define screenshot region

# Constants
distance_threshold = 139  # Adjust as needed
threshold = 0.87  # Template matching threshold

print("Running...")
while True:
    # Take a screenshot of the game region
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_BGR2GRAY)

    # Locate the dinosaur
    dino_res = cv.matchTemplate(screenshot, dino_template, cv.TM_CCOEFF_NORMED)
    dino_loc = np.where(dino_res >= threshold)
    dino_positions = list(zip(*dino_loc[::-1]))  # Convert to a list of (x, y)

    if not dino_positions:
        continue  # Skip iteration if dinosaur is not found

    dino_x, dino_y = dino_positions[0]  # Assume the first match is the dinosaur
    dino_w, dino_h = dino_template.shape[::-1]

    # Check for obstacles
    for cartoon in all_cartoons:
        template = cv.imread(cartoon, 0)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[:6]):
            # Calculate distance between the dinosaur and the obstacle
            distance = pt[0] - (dino_x + dino_w)
            if 0 < distance < distance_threshold:
                # Trigger a space keypress
                pyautogui.press("space")
                print(f"Jump! Distance: {distance}")
                break  # Jump once for the closest obstacle
