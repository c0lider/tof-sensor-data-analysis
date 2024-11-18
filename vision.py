import cv2
import numpy as np

PERSON_SIZE = 100
MOVEMENT_THRESHOLD = 5


def count_people(mask: np.array, person_size=PERSON_SIZE) -> int:
    """
    Count the number of people in a given binary mask image.
    This function takes a binary mask image where the regions of interest (people)
    are represented by 1 and the background is represented by 0.
    It identifies contours in the mask and counts the number of contours that have an area
    greater than a specified person size, which is assumed to correspond to the size of a person.
    Parameters:
        mask (np.array): A binary mask image as a NumPy array.
        person_size (int, optional): The minimum contour area to be considered as a person.
                                    Defaults to PERSON_SIZE.
    Returns:
        int: The number of people detected in the mask.
    """
    mask = mask.astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_count = sum(
        1 for contour in contours if cv2.contourArea(contour) > person_size
    )
    return person_count


def count_and_track_people_direction(
    previous_frame: np.array,
    current_frame: np.array,
    person_size=PERSON_SIZE,
    movement_threshold: int = MOVEMENT_THRESHOLD,
):
    """
    Analyzes two consecutive frames to count and track the direction of people movement.
    This function takes two frames (previous and current) and identifies the contours
    representing people in each frame. It then calculates the centroids of these contours
    and matches them between frames to estimate the movement direction of each person.
    Parameters:
        previous_frame (np.array): The previous frame containing binary mask data.
        current_frame (np.array): The current frame containing binary mask data.
        person_size (int): The minimum contour area to be considered as a person. Default is PERSON_SIZE.
        movement_threshold(int): The minimum amount of change in movement which is neccessary in order to detect a movement
    Returns:
        list of tuples: Each tuple contains:
            - start point (tuple): The centroid of the person in the previous frame.
            - end point (tuple or None): The centroid of the person in the current frame, or None if no match found.
            - direction (str): The direction of movement ('up', 'down', or 'none').
    """
    previous_mask = previous_frame.astype(np.uint8)
    current_mask = current_frame.astype(np.uint8)

    prev_contours, _ = cv2.findContours(
        previous_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    curr_contours, _ = cv2.findContours(
        current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    prev_centroids = [
        cv2.moments(c) for c in prev_contours if cv2.contourArea(c) > person_size
    ]
    curr_centroids = [
        cv2.moments(c) for c in curr_contours if cv2.contourArea(c) > person_size
    ]

    # convert centroids to points
    prev_points = [
        (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        for m in prev_centroids
        if m["m00"] != 0
    ]
    curr_points = [
        (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        for m in curr_centroids
        if m["m00"] != 0
    ]

    directions = []

    for prev in prev_points:
        # ensure there are points in the current frame
        if not curr_points:
            directions.append((prev, None, "none"))
            continue

        # find the closest current point to estimate movement
        closest_curr = min(
            curr_points, key=lambda c: np.linalg.norm(np.array(prev) - np.array(c))
        )
        movement_vector = (closest_curr[0] - prev[0], closest_curr[1] - prev[1])

        # convert movement vector to string
        if abs(movement_vector[1]) > movement_threshold:
            direction_str = "up" if movement_vector[1] < 0 else "down"
        else:
            direction_str = "none"

        # (start point, end point, direction vector)
        directions.append((prev, closest_curr, direction_str))

    return directions
