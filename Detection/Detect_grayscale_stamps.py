## Libraries
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, showAxis=False, size=(20, 10)):
    plt.figure(figsize=size)
    if not showAxis:
        plt.axis("off")
    if len(img.shape) == 3:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img, cmap="gray")


def auto_canny(grayim, sigma=0.93):
    # Find edges using canny edge detector
    # compute the median of the single channel pixel intensities
    v = np.median(grayim)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(grayim, lower, upper)
    # return the edged image
    return edged


def detect(imgFile: str, output_dir=None) -> list:
    if output_dir is None:
        output_dir = Path("temp") / "detection"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    img = cv2.imread(imgFile)
    cv2.imwrite(str(output_dir.joinpath("image.png")), img)

    origin_image = img.copy()
    cv2.imwrite(str(output_dir.joinpath("original_image.png")), origin_image)

    # Blur & detect the edges

    # Convert to Grayscale
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_dir.joinpath("gray.png")), gray)

    # Blur to remove noise
    blur = cv2.bilateralFilter(gray.copy(), 8, 150, 30)
    cv2.imwrite(str(output_dir.joinpath("filter0.png")), blur)

    # Find the edges and display the image
    # sigma_values = [0.01, 0.8, 1.0, 1.2, 1.5]
    # for count, sigma in enumerate(sigma_values):
    #     edged = auto_canny(blur, sigma)
    #     imshow(edged)
    #     cv2.imwrite(folder_path.__add__("\edges{}.png".format(count)), edged)
    edged = cv2.Canny(blur, 150, 70)
    cv2.imwrite(str(output_dir.joinpath("edged.png")), blur)

    ## Find Contours

    # detect the contours on the binary image
    contours, _ = cv2.findContours(
        image=edged.copy(),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    print(f"Total nr of contours found: {len(contours)}")

    top_n = 10
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # print(len(sorted_contours))
    # for s in sorted_contours:
    #     print(cv2.contourArea(s))
    # Filter contours based on area
    filtered_contours = [contour for contour in sorted_contours if 90 < cv2.contourArea(contour) < 17000]

    # Create a blank mask image for the contours
    mask = np.zeros_like(origin_image, dtype=np.uint8)

    # Draw contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Save the mask image
    # cv2.imwrite(os.path.join(folder_path, "contours.png"), mask)

    # Extract the stamps
    files = []
    for j, contour in enumerate(sorted_contours):
        # Find the bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on width and height
        if h > 90 and w > 90 and w < 163 and h < 1180:
            # Extract the stamp region
            stamp = origin_image[y:y + h, x:x + w]
            filepath = str(output_dir.joinpath("extacted{}.png".format(j)))
            files.append(filepath)
            cv2.imwrite(filepath, stamp)
            # Save the extracted stamp
            # print(os.path.basename(img_file))
            cv2.imwrite(os.path.join(output_dir, os.path.basename(imgFile)), stamp)
    return files
    # Sort Contours by Area and get topN
    # topN = 10
    # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # sorted_contours = sorted_contours[:topN]
    # filtered_contours = []
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 17000 and area > 50:
    #         filtered_contours.append(contour)
    # #  Fill the area inside contours
    # filteredCircle = np.zeros((img.shape[:2]), dtype=np.uint8)
    # cv2.drawContours(
    #     image=filteredCircle,
    #     contours=sorted_contours,
    #     contourIdx=-1,
    #     color=(255, 255, 255),
    #     thickness=cv2.FILLED,
    # )
    #
    # cv2.imwrite(str(output_dir.joinpath("contours10.png")), filteredCircle)
    #
    # files = []
    #
    # for i, contour in enumerate(filtered_contours):
    #     # Create a blank mask image for the contour
    #     mask = np.ones_like(origin_image, dtype=np.uint8) * 255
    #
    #     # Draw the contour on the mask
    #     cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    #
    #     # Apply the mask to the original image
    #     result = cv2.bitwise_and(origin_image, mask)
    #
    #     # Find the bounding rectangle around the contour
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w > 40 and h > 40:
    #         # Extract the contour as a separate image
    #         extracted_image = result[y: y + h, x: x + w]
    #
    #         # Save the extracted image
    #         filepath = str(output_dir.joinpath("extacted{}.png".format(i)))
    #         files.append(filepath)
    #         cv2.imwrite(filepath, extracted_image)
    #
    # kernel = np.ones((3, 3), np.uint8)
    # closedCircle = cv2.morphologyEx(
    #     filteredCircle, cv2.MORPH_CLOSE, kernel, iterations=1
    # )
    # # imshow(closedCircle)
    # cv2.imwrite(str(output_dir.joinpath("closedCircle.png")), closedCircle)
