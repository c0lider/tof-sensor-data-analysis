"""
This module provides utility functions for loading, processing, and visualizing depth data from binary files.

Functions:
- load_input_file_paths: Loads the file paths of input data files from a specified directory.
- load_depth_data_sets: Loads depth data sets from a list of binary files.
- load_depth_data_frames: Loads depth data frames from a binary file.
- create_gif_from_frames: Creates a GIF from a sequence of frames and saves it to a specified file.
- get_absolute_path_and_mkdirs: Ensures the directory for a given file path exists and returns the absolute path.

Constants:
- DEFAULT_FRAME_WIDTH: The default width of each frame.
- DEFAULT_FRAME_HEIGHT: The default height of each frame.
- DATA_DIR_NAME: The name of the directory containing the input data files.

Dependencies:
- os: Provides a way of using operating system dependent functionality.
- numpy: A package for scientific computing with Python.
- matplotlib.pyplot: A plotting library for creating static, animated, and interactive visualizations.
- PIL: Python Imaging Library, adds image processing capabilities.
- typing: Provides runtime support for type hints.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from typing import List


DEFAULT_FRAME_WIDTH = 80
DEFAULT_FRAME_HEIGHT = 60
DATA_DIR_NAME = "Messdaten"


def load_input_file_paths(input_data_dir: str = DATA_DIR_NAME) -> List[str]:
    """
    Loads the file paths of input data files from the specified directory.

    This function searches for files with a ".bin" extension in the given directory.
    If the directory does not exist or is empty, the function will print an error message and exit.
    If a file does not exist or does not have the correct file type, it will be skipped.

    Args:
        input_data_dir (str): The directory to search for input data files. Defaults to DATA_DIR_NAME.

    Returns:
        List[str]: A list of absolute file paths to the input data files.

    Raises:
        SystemExit: If the directory does not exist or is empty.
    """
    if os.path.isabs(input_data_dir):
        data_abs_path = os.path.join(os.path.dirname(__file__), input_data_dir)
    else:
        data_abs_path = input_data_dir

    if not os.path.exists(data_abs_path):
        print(f"Directory {data_abs_path} does not exist")
        exit(1)

    file_names = []

    for file_name in os.listdir(data_abs_path):
        abs_file_path = os.path.join(data_abs_path, file_name)

        if not os.path.isfile(abs_file_path):
            print(f"Could not load file {abs_file_path} because it does not exist")
            continue

        if not os.path.splitext(file_name)[1] == ".bin":
            print(
                f"Skipping file {file_name} because it doesn't have the correct file type"
            )
            continue

        file_names.append(abs_file_path)

    if not len(file_names):
        print(f"Input directory {data_abs_path} is empty")
        exit(1)
    else:
        print(f"Found {len(file_names)} input files in {data_abs_path}")

    return file_names


def load_depth_data_sets(
    file_names: List[str],
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
) -> dict[np.ndarray]:
    """
    Loads depth data sets from a list of binary files.

    This function reads binary files containing depth data and converts them into a dictionary of numpy arrays.
    Each key in the dictionary is the file name (with spaces replaced by underscores and the extension changed to ".gif"),
    and the corresponding value is a numpy array containing the depth data frames.

    Args:
        file_names (List[str]): A list of file paths to the binary files.
        width (int, optional): The width of each frame. Defaults to DEFAULT_FRAME_WIDTH.
        height (int, optional): The height of each frame. Defaults to DEFAULT_FRAME_HEIGHT.
    Returns:
        dict[np.ndarray]: A dictionary where the keys are file names and the values are numpy arrays of depth data frames.
    """
    data_sets = {}

    for file_name in file_names:
        output_file_name = (
            os.path.splitext(file_name.split("/")[-1])[0] + ".gif"
        ).replace(" ", "_")
        try:
            data_sets[output_file_name] = load_depth_data_frames(
                file_name, width, height
            )
        except ValueError as e:
            print(f"{file_name}: {e}")
            continue

    return data_sets


def load_depth_data_frames(
    file_name: str, width: int = DEFAULT_FRAME_WIDTH, height: int = DEFAULT_FRAME_HEIGHT
) -> np.ndarray:
    """
    Loads depth data frames from a binary file.
    This function reads depth data from a binary file and reshapes it into a 3D numpy array
    with the specified width and height for each frame. The data is expected to be in
    float64 format.
    Args:
        file_name (str): The path to the binary file containing the depth data.
        width (int, optional): The width of each frame. Defaults to DEFAULT_FRAME_WIDTH.
        height (int, optional): The height of each frame. Defaults to DEFAULT_FRAME_HEIGHT.
    Returns:
        np.ndarray: A 3D numpy array of shape (num_frames, height, width) containing the
        depth data frames. If the file is not found, an empty array is returned.
    Raises:
        ValueError: If the total size of the data is not divisible by the product of width
        and height, indicating an incorrect frame size.
    """
    try:
        with open(file_name, "rb") as file:
            data = np.fromfile(file, dtype=np.float64)
    except FileNotFoundError as e:
        print(e)
        return np.array([])

    if data.size % (width * height) != 0:
        raise ValueError(f"Incorrect frame size {width} x {height}")

    return data.reshape(-1, height, width)


# TODO: find the correct vmax
def create_gif_from_frames(
    frames: np.array, output_path: str, vmin=0, vmax=3321
) -> str:
    """
    Creates a GIF from a sequence of frames and saves it to the specified output path.
    Parameters:
    Args:
        frames (np.array): A numpy array of frames where each frame is a 2D array representing an image.
        output_path (str): The path where the resulting GIF will be saved.
        vmin (int, optional): The minimum data value that corresponds to colormap "hot". Default is 0.
        vmax (int, optional): The maximum data value that corresponds to colormap "hot". Default is 3321.
    Returns:
        str: The absolute path to the saved GIF file.
    Raises:
        ValueError: If the provided frames do not contain any data.
        IOError: If there is an error while saving the GIF file.
    Notes:
    - The function uses the "hot" colormap for visualizing the frames.
    - The function ensures that the output directory exists before saving the GIF.
    - The duration of each frame in the GIF is set to 200 milliseconds.
    - The GIF is set to loop indefinitely.
    """
    if frames.shape[0] == 0:
        raise ValueError("The provided frames do not contain any data")

    image_frames = []

    for i in range(frames.shape[0]):
        plt.imshow(frames[i], cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.axis("off")

        plt.tight_layout()
        plt.draw()

        image = Image.frombytes(
            "RGB", plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb()
        )
        image_frames.append(image)

        plt.clf()

    output_path = get_absolute_path_and_mkdirs(output_path)

    try:
        image_frames[0].save(
            output_path,
            format="GIF",
            append_images=image_frames[1:],
            save_all=True,
            duration=100,
            loop=0,
            optimize=True,
        )

        return output_path
    except IOError as e:
        print(f"There was an error while saving {output_path}:")
        print(e)


def get_absolute_path_and_mkdirs(file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    file_dir = os.path.dirname(file_path)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    return file_path


## TODO: error handling
def add_text_to_gif(file_path: str):
    print("asdf")
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(file_path) as img:
        frames_with_text = []

        for frame in range(img.n_frames):
            img.seek(frame)
            frame_copy = img.copy().convert("RGBA")

            draw = ImageDraw.Draw(frame_copy)
            text = f"Frame {frame + 1}"
            font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )

            text_position = (
                (frame_copy.width - text_width) // 2,
                frame_copy.height - text_height - 10,
            )
            draw.text(text_position, text, font=font, fill="black")

            frames_with_text.append(frame_copy)

        frames_with_text[0].save(
            file_path,
            save_all=True,
            append_images=frames_with_text[1:],
            duration=100,
            loop=0,
        )
