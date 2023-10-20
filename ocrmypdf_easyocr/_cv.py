# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""OpenCV algorithms."""

from __future__ import annotations

import cv2 as cv


def detect_skew(img) -> float:
    """Detect the skew angle of an image.

    Implements the algorithm described in:
    https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=5)
    contours, _hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_contour = contours[0]
    rect = cv.minAreaRect(largest_contour)

    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return angle
