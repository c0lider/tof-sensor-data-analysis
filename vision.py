import cv2
import numpy as np

PERSON_SIZE = 100


def count_people(mask: np.array, person_size=PERSON_SIZE) -> int:
    mask = mask.astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_count = sum(
        1 for contour in contours if cv2.contourArea(contour) > person_size
    )
    return person_count
