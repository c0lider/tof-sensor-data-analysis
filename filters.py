import numpy as np

# from scipy import stats


def compute_background(frames: np.array) -> np.array:
    # Alternative background detection methods
    # mode_background, _ = stats.mode(frames, axis=0)
    # return mode_background[0]

    # return np.median(frames, axis=0)

    return np.mean(frames, axis=0)


def subtract_background_from_frames(
    frames: np.array, background: np.array = None, threshold: int = 100
) -> np.array:
    if background is None:
        background = compute_background(frames)

    foreground = []

    for frame in frames:
        foreground.append(subtract_background_from_frame(frame, background, threshold))

    return np.stack(foreground)


def subtract_background_from_frame(
    frame: np.array, background: np.array, threshold: int = 50
) -> np.array:
    foreground = np.abs(frame - background)
    return (foreground > threshold).astype(np.float64)
