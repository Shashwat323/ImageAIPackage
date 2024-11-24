import cv2
import itertools
import numpy as np
import random
from PIL import Image


def gray_scale(image_file_path: str, mode: int = 1):
    """
    Convert an image to grayscale.

    Parameters:
    - image_file_path: str : Path to the image file.
    - mode: int :
        - 1: Gray scale 0-255
        - 2: 8-bit gray scale
        - 3: Black and white

    Returns:
    - Image object : The gray scaled image.
    """
    image = Image.open(image_file_path)
    if mode == 1:
        gray_scaled_image = image.convert("L")
    elif mode == 2:
        gray_scaled_image = image.quantize(colors=256)
        gray_scaled_image = image.convert("L")
    elif mode == 3:
        gray_scaled_image = image.convert("1")
    else:
        print("Error, incorrect mode")
    return gray_scaled_image


def save_image(img: np.ndarray, output_file_path: str) -> None:
    """
    Save a numpy array as an image file.

    Parameters:
    - img: np.ndarray : The image array to be saved.
    - output_file_path: str : Path to save the image file.
    """
    if img is None:
        raise ValueError("Cannot save an image that is None.")

    # Save the image
    cv2.imwrite(output_file_path, img)


def image_path_to_np_array(image_file_path: str) -> np.ndarray:
    """
    Convert an image file to a numpy array.

    Parameters:
    - image_file_path: str : Path to the image file.

    Returns:
    - np.ndarray : The image as a numpy array.
    """
    img = cv2.imread(image_file_path)
    return img


def crop(img: np.ndarray, new_width: int = float('inf')) -> np.ndarray:
    """
    Crop an image to the given width or the image's smallest dimension, whichever is smaller.

    Parameters:
    - img: np.ndarray : The image to be cropped.
    - new_width: int : The new width to which the image should be cropped.

    Returns:
    - np.ndarray : The cropped image.
    """
    height, width = img.shape[:2]
    size = min(height, width, new_width)
    x = int((width - size) / 2)
    y = int((height - size) / 2)
    return img[y:y + size, x:x + size]


def resize(img: np.ndarray, new_width: int = 224) -> np.ndarray:
    """
    Resize an image to the given width while maintaining aspect ratio.

    Parameters:
    - img: np.ndarray : The image to be resized.
    - new_width: int : The new width for the image.

    Returns:
    - np.ndarray : The resized image.
    """
    return cv2.resize(img, (new_width, new_width))


def region_grow(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Apply the region-growing algorithm to an image.

    Parameters:
    - img: np.ndarray : The input image.
    - threshold: int : The threshold value for the region-growing algorithm.

    Returns:
    - np.ndarray : The segmented image.
    """
    regiongrow = RegionGrow(img, threshold)
    return regiongrow.apply_region_grow()


class Stack:
    """A simple stack implementation."""

    def __init__(self):
        self.item = []
        self.obj = []

    def push(self, value):
        """Push an item onto the stack."""
        self.item.append(value)

    def pop(self):
        """Pop an item off the stack."""
        return self.item.pop()

    def size(self) -> int:
        """Return the size of the stack."""
        return len(self.item)

    def isEmpty(self) -> bool:
        """Check if the stack is empty."""
        return self.size() == 0

    def clear(self):
        """Clear the stack."""
        self.item = []


class RegionGrow:
    """Class to perform region growing on an image."""

    def __init__(self, img, th):
        """
        Initialize the RegionGrow object.

        Parameters:
        - img: np.ndarray : The input image.
        - th: float : The threshold value for region growing.
        """
        self.read_image(img)
        self.h, self.w, _ = self.im.shape
        self.passedBy = np.zeros((self.h, self.w), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def read_image(self, img):
        """
        Read and process the input image.

        Parameters:
        - img: np.ndarray : The input image.
        """
        self.im = img.astype('int')

    def get_neighbour(self, x0, y0):
        """
        Get the neighboring pixels of a given pixel.

        Parameters:
        - x0: int : The x-coordinate of the pixel.
        - y0: int : The y-coordinate of the pixel.

        Returns:
        - list : The list of neighboring pixels.
        """
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.boundaries(x := x0 + i, y := y0 + j)
        ]

    def create_seeds(self):
        """
        Create initial seed points for region growing.

        Returns:
        - list : The list of seed points.
        """
        return [
            [self.h / 2, self.w / 2],
            [self.h / 3, self.w / 3], [2 * self.h / 3, self.w / 3], [self.h / 3 - 10, self.w / 3],
            [self.h / 3, 2 * self.w / 3], [2 * self.h / 3, 2 * self.w / 3], [self.h / 3 - 10, 2 * self.w / 3],
            [self.h / 3, self.w - 10], [2 * self.h / 3, self.w - 10], [self.h / 3 - 10, self.w - 10]
        ]

    def apply_region_grow(self) -> np.ndarray:
        """
        Apply region growing algorithm to the image.

        Returns:
        - np.ndarray : The segmented image.
        """
        randomseeds = self.create_seeds()
        np.random.shuffle(randomseeds)

        for x0 in range(self.h):
            for y0 in range(self.w):

                if self.passedBy[x0, y0] == 0:  # and (np.all(self.im[x0,y0] > 0)) :
                    self.currentRegion += 1
                    self.passedBy[x0, y0] = self.currentRegion
                    self.stack.push((x0, y0))
                    self.prev_region_count = 0

                    while not self.stack.isEmpty():
                        x, y = self.stack.pop()
                        self.bfs(x, y)
                        self.iterations += 1

                    if self.passed_all():
                        break

                    if self.prev_region_count < 8 * 8:
                        x0, y0 = self.reset_region(x0, y0)

        [self.color_pixel(i, j) for i, j in itertools.product(range(self.h), range(self.w))]
        return self.SEGS

    def reset_region(self, x0, y0):
        """
        Reset the current region during region growing.

        Parameters:
        - x0: int : The x-coordinate of the region.
        - y0: int : The y-coordinate of the region.

        Returns:
        - tuple : The reset x and y coordinates.
        """
        self.passedBy[self.passedBy == self.currentRegion] = 0
        x0 = random.randint(x0 - 4, x0 + 4)
        y0 = random.randint(y0 - 4, y0 + 4)
        x0 = np.clip(x0, 0, self.h - 1)
        y0 = np.clip(y0, 0, self.w - 1)
        self.currentRegion -= 1
        return x0, y0

    def color_pixel(self, i, j):
        """
        Color a pixel based on its region.

        Parameters:
        - i: int : The x-coordinate of the pixel.
        - j: int : The y-coordinate of the pixel.
        """
        val = self.passedBy[i][j]
        self.SEGS[i][j] = (255, 255, 255) if (val == 0) else (val * 35, val * 90, val * 30)

    def bfs(self, x0, y0):
        """
        Perform the Breadth-First Search (BFS) for region growing.

        Parameters:
        - x0: int : The starting x-coordinate.
        - y0: int : The starting y-coordinate.
        """
        regionNum = self.passedBy[x0, y0]
        elems = [np.mean(self.im[x0, y0])]
        var = self.thresh
        neighbours = self.get_neighbour(x0, y0)

        for x, y in neighbours:
            if self.passedBy[x, y] == 0 and self.distance(x, y, x0, y0) < var:
                if self.passed_all():
                    break

                self.passedBy[x, y] = regionNum
                self.stack.push((x, y))
                elems.append(np.mean(self.im[x, y]))
                var = np.var(elems)
                self.prev_region_count += 1
            var = max(var, self.thresh)

    def passed_all(self, max_iteration=200000):
        """
        Check if all pixels have been processed or max iterations reached.

        Parameters:
        - max_iteration: int : The maximum number of iterations allowed.

        Returns:
        - bool : True if all pixels have been processed or max iterations reached; False otherwise.
        """
        return self.iterations > max_iteration or np.all(self.passedBy > 0)

    def boundaries(self, x, y):
        """
        Check if a pixel is within image boundaries.

        Parameters:
        - x: int : The x-coordinate.
        - y: int : The y-coordinate.

        Returns:
        - bool : True if the pixel is within boundaries; False otherwise.
        """
        return 0 <= x < self.h and 0 <= y < self.w

    def distance(self, x, y, x0, y0) -> float:
        """
        Calculate the Euclidean distance between two pixels.

        Parameters:
        - x: int : The x-coordinate of the first pixel.
        - y: int : The y-coordinate of the first pixel.
        - x0: int : The x-coordinate of the second pixel.
        - y0: int : The y-coordinate of the second pixel.

        Returns:
        - float : The Euclidean distance between the two pixels.
        """
        return np.linalg.norm(self.im[x0, y0] - self.im[x, y])
