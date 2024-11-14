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

from PIL import Image, ImageDraw, ImageFont

from typing import List


DEFAULT_FRAME_WIDTH = 80
DEFAULT_FRAME_HEIGHT = 60
DATA_DIR_NAME = "Messdaten"
DEFAULT_POINT_RADIUS = 10
SCALING_FACTOR = 8


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


def create_gif_from_frames(frames: np.array, output_path: str) -> str:
    """
    Creates a GIF from a sequence of frames and saves it to the specified output path.
    Parameters:
    Args:
        frames (np.array): A numpy array of frames where each frame is a 2D array representing an image.
        output_path (str): The path where the resulting GIF will be saved.
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

    vmin = np.min(frames)
    vmax = np.max(frames)

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


def get_absolute_path_and_mkdirs(file_path: str) -> str:
    """
    Converts a relative file path to an absolute path and creates any necessary directories.

    This function takes a file path as input and performs the following steps:
    1. Checks if the provided file path is an absolute path. If not, it converts it to an absolute path.
    2. Extracts the directory part of the file path.
    3. Checks if the directory exists. If it does not exist, it creates the directory (including any necessary parent directories).

    Args:
        file_path (str): The file path to be converted and checked.

    Returns:
        str: The absolute file path.
    """

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    file_dir = os.path.dirname(file_path)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    return file_path


def add_text_to_gif(file_path: str, caption_per_frame: List[str] | str) -> None:
    """
    Adds text captions to each frame of a GIF file.
    Parameters:
        file_path (str): The path to the GIF file to which text will be added.
        caption_per_frame (List[str] | str): A list of captions for each frame or a single string to be used as a caption for all frames.
    Returns:
        None
    Notes:
    - If `caption_per_frame` is a string, the same caption will be added to all frames.
    - If `caption_per_frame` is a list, each caption will be added to the corresponding frame.
    - The function prints an error message if the number of frames in the GIF does not match the number of captions provided.
    """
    try:
        with Image.open(file_path) as img:
            simple_caption = False

            # caption_per_frame is a string => display it on all frames
            if isinstance(caption_per_frame, str):
                simple_caption = True
            # TODO: check indices
            elif abs(img.n_frames - len(caption_per_frame)) > 1:
                print(
                    f"The amount of frames {img.n_frames} does not match the amount of captions {len(caption_per_frame)} provided. Caption was not added"
                )
                return

            frames_with_text = []

            for frame_no in range(img.n_frames):
                img.seek(frame_no)
                frame_copy = img.copy().convert("RGBA")
                draw = ImageDraw.Draw(frame_copy)

                if simple_caption:
                    text = caption_per_frame
                else:
                    text = caption_per_frame[frame_no]
                font = ImageFont.load_default()

                draw.text((10, 10), text, font=font, fill="black")

                frames_with_text.append(frame_copy)

    except FileNotFoundError:
        print("The specified file does not exist.")

    try:
        frames_with_text[0].save(
            file_path,
            save_all=True,
            append_images=frames_with_text[1:],
            duration=100,
            loop=0,
        )
    except OSError as e:
        print("Could not save file: ", e)


def add_points_to_gif(
    file_path: str, points: List[int], point_radius: int = DEFAULT_POINT_RADIUS
):
    """
    Adds points to each frame of a GIF image and saves the modified GIF.
    Parameters:
        file_path (str): The path to the GIF file.
        points (List[int]): A list of points to be added to each frame. Each element in the list should be a list of tuples,
                            where each tuple represents the (x, y) coordinates of a point.
        point_radius (int, optional): The radius of the points to be drawn. Defaults to DEFAULT_POINT_RADIUS.
    Returns:
        None
    Notes:
        - The function checks if the number of frames in the GIF matches the number of point data provided. If they do not match,
        the function prints a message and does not add the points.
        - The points are scaled by a SCALING_FACTOR before being drawn on the frames.
        - The modified GIF is saved with a duration of 100ms per frame and loops indefinitely.
    """
    try:
        with Image.open(file_path) as img:
            # TODO: check indices
            if abs(img.n_frames - len(points)) > 1:
                print(
                    f"The amount of frames {img.n_frames} does not match the amount of point data {len(points)} provided. Points were not added"
                )
                return

            frames_with_points = []

            for frame_no in range(img.n_frames):
                img.seek(frame_no)
                frame_copy = img.copy().convert("RGBA")
                draw = ImageDraw.Draw(frame_copy)

                for person in points[frame_no]:
                    draw.ellipse(
                        (
                            person[0] * SCALING_FACTOR - point_radius,
                            person[1] * SCALING_FACTOR - point_radius,
                            person[0] * SCALING_FACTOR + point_radius,
                            person[1] * SCALING_FACTOR + point_radius,
                        ),
                        fill="red",
                        outline="black",
                    )

                frames_with_points.append(frame_copy)

    except FileNotFoundError:
        print("The specified file does not exist.")

    try:
        frames_with_points[0].save(
            file_path,
            save_all=True,
            append_images=frames_with_points[1:],
            duration=100,
            loop=0,
        )
    except OSError as e:
        print("Could not save file: ", e)
