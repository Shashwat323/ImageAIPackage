import PIL
import cv2
import itertools
import numpy as np
import random
import rembg
from typing import Callable, List, Union
from PIL import Image, ImageOps, ImageEnhance
import onnxruntime

def overlay_images(base_image_array: np.ndarray, overlay_image_array: np.ndarray, transparency: float = 1.0):
    if not (0.0 <= transparency <= 1.0):
        raise ValueError('Transparency must be between 0.0 and 1.0')
    base_image = Image.fromarray(base_image_array).convert('RGBA')
    overlay_image = Image.fromarray(overlay_image_array).convert('RGBA')
    if base_image.size != overlay_image.size:
        raise ValueError('Base image and overlay image must be the same size')
    overlay_image = Image.blend(Image.new("RGBA", overlay_image.size, (0, 0, 0, 0)), overlay_image, transparency)
    result = Image.alpha_composite(base_image, overlay_image)
    return np.array(result)

def adjust_contrast(image_array: np.ndarray, mini: float = 0.5, maxi: float = 1.5):
    image = Image.fromarray(image_array)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(random.uniform(mini, maxi))
    return np.array(enhanced_image)

def adjust_hue(image_array: np.ndarray, mini: float = 0.7, maxi: float = 1.3):
    image = Image.fromarray(image_array).convert("HSV")
    hsv_array = np.array(image)
    for i in range(hsv_array.shape[0]):  # height
        for j in range(hsv_array.shape[1]):  # width
            h, s, v = hsv_array[i, j]  # get hue, saturation, value
            h = int(h)
            h = (h * random.uniform(mini, maxi)) % 255
            hsv_array[i, j] = [h, s, v]
    adjusted_image = Image.fromarray(hsv_array, mode="HSV").convert("RGB")
    return np.array(adjusted_image)

def adjust_brightness(image_array: np.ndarray, mini: float = 0.7, maxi: float = 1.3):
    image = Image.fromarray(image_array)
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(random.uniform(mini, maxi))
    return np.array(enhanced_image)

def square_rotate(image_array: np.ndarray):
    image = Image.fromarray(image_array)
    angle = random.choice([90, 180, 270])
    rotated_image = image.rotate(angle)
    return np.array(rotated_image)

def mirror_image(image_array: np.ndarray):
    roll = random.random()
    image = Image.fromarray(image_array)
    if roll <= 0.5:
        mirrored_image = ImageOps.mirror(image)
    else:
        mirrored_image = image
    return np.array(mirrored_image)

def random_rotate(image_array: np.ndarray):
    angle = random.randint(0, 360)
    image = Image.fromarray(image_array)
    rotated_image = image.rotate(angle, expand=True, fillcolor=(0, 255, 0))
    return np.array(rotated_image)

def random_crop(image_array: np.ndarray):
    image = Image.fromarray(image_array)
    width, height = image.size
    smallest_edge = width if width < height else height
    x_min = random.randint(0, width - smallest_edge) if width > smallest_edge else 0
    y_min = random.randint(0, height - smallest_edge) if height > smallest_edge else 0
    box = (x_min, y_min, x_min + smallest_edge, y_min + smallest_edge)
    cropped_image = image.crop(box)
    return np.array(cropped_image)


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

# Remove background.
# removeBackground(): Remove background of an image.

# Adjust the contrast in an image.
# histogramEqualisation() : Adjust contrast in an image using histogram equalisation

# There are four methods of normalising images.
# These will all convert the images to greyscale 
# normalise() : Between 0 and 1
# zeroMeanOneVar() : Z-Score of each pixel
# minMaxScaling() : rescale pixel values to fit in a set range (0-1 by default)
# meanNormalisation() : normalise based on the mean pixel value

def removeBackground(img: np.ndarray):
    """
    Remove background.
    Area outside will become white
    Parameter:
    - img (np.ndarray): cv2 image array in BGR colour. (cv2.imread(path_to_image_file))
    Output:
    - output (np.ndarray): Image with white background in array form
    """
    # Remove background
    output = rembg.remove(img)
    return output

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def histogramEqualisation(img: np.ndarray, greyscale: bool=False, clahe: bool=False, gridsize: int=8):
    """
    Adjust the contrast in an image using histogram equalisation.
    Parameters:
    - img (np.ndarray): cv2 image array in BGR colour. (cv2.imread(path_to_image_file))
    - greyscale (bool): Set the output to be greyscale (default False)
    - clahe (bool): Use Contrast Limited Adaptive Histogram Equalization (CLAHE) (default False)
    - gridsize (int): gridsize of CLAHE to be applied (default 8).
    Output:
    - image (np.ndarray): output image in array form. 
    """
    if (clahe == False):
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalisation
            image = cv2.equalizeHist(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    else:
        if (greyscale):
            # Image to grayscale
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create and apply CLAHE
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(grey_img)
        else:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Histogram equalisation on V-channel
            img_hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize)).apply(img_hsv[:, :, 2])
            # Convert to RGB
            image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return image

def normalise(img: np.ndarray,greyscale: bool=False):
    """
    Normalise pixel values to between 0 and 1.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Normalize grayscale
    normalized_gray_image = cv2.normalize(grey_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return normalized_gray_image

def zeroMeanOneVar(img: np.ndarray,greyscale: bool=False):
    """
    Get Z-score of pixel values.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(grey_img)
    # Normalise image
    normalized_image = (grey_img - mean) / std_dev
    return normalized_image

def minMaxScaling(img: np.ndarray,greyscale: bool=False,min: int=0, max: int=1):
    """
    Normalise pixel values by rescaling them to within a range (default to 0-1 inclusive).
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    - min (int): Minimum value of the normalised data (OPTIONAL)
    - max (int): Maximum value of the normalised data (OPTIONAL)
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Min-Max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img)
    # Normalise image
    return min + ((grey_img - min_val) * (max - min)) / (max_val - min_val)

def meanNormalisation(img: np.ndarray,greyscale: bool=False):
    """
    Normalise pixel values based on the mean.
    This will convert the image to greyscale
    Parameters:
    - img (np.ndarray): cv2 image array. (cv2.imread(path_to_image_file))
    - greyscale (bool): Is the image already in greyscale (default False).
    Output:
    - normalized_grey_image (np.ndarray): Normalised image in array form (greyscale)
    """
    if (greyscale == False):
        # Image to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # Find mean and std
    mean, std_dev = cv2.meanStdDev(grey_img)
    # Min-Max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img)
    # Normalize grayscale
    mean_normalized = (grey_img - mean) / (max_val - min_val)
    return mean_normalized

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


def blur(img_input, kernel_size=3):
    """
        blur is a standard blur in which the current pixels value is changed to be the average of it's nxn neighbours

    Args:
        img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
        kernel_size(int): Is the size of the kernel used for blurring

    Returns:
        np.ndarray: The blurred image is returned as a np.ndarray

    Raises:
        ValueError: If kernel_size is not an int, or the img is not inputted as a file path or np.ndarray

    """
    img = img_to_numpy_array(img_input, grey=True)
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an int")
    return cv2.blur(img, (kernel_size, kernel_size))
def bilateral_blur(img_input, diameter=9, sigma_color=75, sigma_space=75):
    """
    bilateral_blur is used to maintain edges whilst smoothing the image

    Args:
        img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
        diameter(int): Size of the kernel
        sigma_color(float): The number of colors to consider in the pixel range
        sigma_space(float): The space between the pixel and neighbouring pixels, higher value means pixels further out impact
                        the pixel value more

    Returns:
        np.ndarray: The blurred image is returned as a np.ndarray

    Raises:
        ValueError: If diameter is not an int, sigma_color or sigma_space are not a numeric value, also img_input

    """
    img = img_to_numpy_array(img_input, grey=True)
    if not isinstance(diameter, int) or not isinstance(sigma_space, int) or not isinstance(sigma_color, int):
        raise ValueError("diameter, sigma_color and sigma_space must be an int")
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
def gaussian_blur(img_input, kernel_size=3, sigma=3):
    """
       gaussian_blur is used to favour closer neighbours more then further neighbours

       Args:
           img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
           kernel_size(int): Size of the kernel used for blurring
           sigma(float): The variance in the values

       Returns:
           np.ndarray: The blurred image is returned as a np.ndarray

       Raises:
           ValueError: If kernel_size is not an int or sigma is not a float, also img_input

       """
    img = img_to_numpy_array(img_input)
    if not isinstance(kernel_size, int) or not (isinstance(sigma, int) or isinstance(sigma, float)):
        raise ValueError("kernel_size must be an int and sigma must be a numeric value")
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
def median_blur(img_input, kernel_size=3):
    """
           median_blur is used to make the current pixel value the median value in all neighbours

           Args:
               img_input(str/np.ndarray): Can be input as a np.ndarray or a file path, is used as the image to be blurred
               kernel_size(int): Size of the kernel used for blurring

           Returns:
               np.ndarray: The blurred image is returned as a np.ndarray

           Raises:
               ValueError: If kernel_size is not an int, also img_input

           """
    img = img_to_numpy_array(img_input, grey=True)
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an int")
    return cv2.medianBlur(img, kernel_size)


def img_to_numpy_array(image_input, grey=False):
    """
           img_to_numpy_array checks whether the input is a file path or a np.ndarray, if it's a file path then it
           gets converted to a np.ndarray then returned, if it's already a np.ndarray then gets returned as is.
           If it's an invalid value then a ValueError is raised and nothing is returned

           Args:
               image_input(str/np.ndarray): The image to be used

           Returns:
               np.ndarray: The image is returned as a np.ndarray

           Raises:
               ValueError: If the image_input is not a valid file path or np.ndarray

           """
    if isinstance(image_input, np.ndarray):
        if grey:
            return convert_to_grey(image_input)
        return image_input
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
        if isinstance(img, np.ndarray):
            if grey:
                return convert_to_grey(img)
            return img
    elif isinstance(image_input, PIL.Image.Image):
        return np.array(image_input)
    if isinstance(image_input, str):
        raise ValueError("File path must be a valid file path")
    raise ValueError("Image must be either a valid file path or a numpy array")


def convert_to_grey(img):
    """
            convert_to_gray takens in a np.ndarray and converts it to greyscale, it is useful because it is not
            necessary to know whether the image being input is already greyscale of not when using the function

            Args:
                img(np.ndarray): The numpy array to be converted
            Returns:
                np.ndarray: A np.ndarray with 2 channels

    """
    """Only takes in np arrays, useful because it checks if image is already greyscale"""
    if len(np.shape(img)) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def laplacian(img_input, kernel_size):
    """
            Laplacian takes the second derivative in the change of pixels to try and find egdes

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image
                kernel_size(int): Takes an int n to determine the nxn size of the kernel

            Returns:
                np.ndarray: Returns a np.ndarray after the laplacian operator has been applied to show edges

            Raises:
                ValueError: Raises a ValueError if the kernel_size is not an integer

    """
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an integer")
    img = img_to_numpy_array(img_input, grey=True)
    dst = cv2.Laplacian(gaussian_blur(img, 3, 1), cv2.CV_16S, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst
def custom_kernel_blur(img_input, kernel):
    """
            Takes a customer kernel as an input and returns a modified version of the image based on the blur

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image
                kernel(np.ndarray): Takes a np.ndarray representing a kernel

            Returns:
                np.ndarray: Returns a np.ndarray that has had the kernel operation applied

            Raises:
                ValueError: Raises a ValueError if kernel is not a np.ndarray


    """
    img = img_to_numpy_array(img_input)
    if not isinstance(kernel, np.ndarray):
        raise ValueError("kernel must be a np.ndarray")
    return cv2.filter2D(img, -1, kernel)

def sharpen(img_input, sharpness = 9):
    """
            Sharpens the image using a custom sharpness

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image
                sharpness(float): Takes a float value representing how much to sharpen the image

            Returns:
                np.ndarray: Returns a np.ndarray representing the sharpened image

            Raises:
                ValueError: Raises a Value Error if sharpness is not a numeric value

    """
    if not (isinstance(sharpness, float) or isinstance(sharpness, int)):
        raise ValueError("sharpness must be a numeric value")
    contrast = (sharpness * (8/9)) / 8 * -1
    img = img_to_numpy_array(img_input)
    return custom_kernel_blur(img, np.array([[contrast, contrast, contrast],
                                             [contrast, sharpness, contrast],
                                             [contrast, contrast, contrast]]))

def adaptive_thresholding(img_input, block_size=11, const_c=2):
    """
            Adaptive Thresholding takes the area around each pixel (block_size) and thresholds the current pixel based on that area
            This allows for better thresholding as it accounts for shadows and other noise

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image
                block_size(int): The block size of the neighbourhood to look at
                const_c(int): const_c is subtracted from the weighted sum of the neighbourhood pixels

            Returns:
                np.ndarray: Returns a np.ndarray of the thresholded image
            
            Raises:
                ValueError: Raises a ValueError if block_size is not an int or const_c isn't an int

    """
    if not isinstance(block_size, int) or not isinstance(const_c, int):
        raise ValueError("block_size and const_c must both be ints")

    img = img_to_numpy_array(img_input, grey=True)
    blur = median_blur(img, 5)
    return cv2.adaptiveThreshold(median_blur(blur, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, const_c)

def otsus_binarization_thresholding(img_input):
    """
            Otsus binariation threshoding looks for two classes that either maximise inter-class variance
            or minimsize intra-class variance

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image

            Returns:
                np.ndarray: A np.ndarray representing the result of otsus binrization thresholding

    """
    img = img_to_numpy_array(img_input, grey=True)
    blur = gaussian_blur(img, 5, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def edge_detection(img_input, min_val=100, max_val=200):
    """
            Uses canny edge detector to outline edges in the image

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image
                min_val(int): Takes an integer to represent min_val for hysterisis thresholding
                max_val(int): Takes an integer to represent max_val for hysterisis thresholding

            Returns:
                np.ndarray: A numpy array representing the input image with the edges drawn on it

            Raises:
                ValueError: If min_val or max_val are not integers, as well as a value error
                if min_val is greater or equal to max_val
    """
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise ValueError("min_val and max_val must be input as integers")
    if (min_val < 0 or min_val > 255) or (max_val < 0 or max_val > 255) or (max_val - min_val <= 0):
        raise ValueError("min_val and max_val must be between 0 and 255 and min_val cannot be greater or equal to max_val")
    img = img_to_numpy_array(img_input, grey=True)
    return cv2.Canny(img, min_val, max_val)

def watershed(img_input):
    """
            Watershed takes advantage of the fact that any greyscale image can be viewed as a topographic map
            The algorithm fills the valleys, the problem with this is that as valleys water level rises,
            they may merge with other valleys, the algorithm would then build a barrier (a dam) between these valleys and this will
            be the segmentation between valleys

            Args:
                img_input(np.ndarray/str): Takes a np.ndarray or a file path to an image

            Returns:
                np.ndarray: Returns a np.ndarray with the borders of the valleys

    """
    returnImg = img_to_numpy_array(img_input)
    img = img_to_numpy_array(img_input, grey=True)

    img = otsus_binarization_thresholding(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_background = cv2.dilate(img, kernel, iterations=2)
    distance = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    _, sure_foreground = cv2.threshold(distance, 0.5 * distance.max(), 255, cv2.THRESH_BINARY)
    sure_foreground = sure_foreground.astype(np.uint8)

    unknown = cv2.subtract(sure_background, sure_foreground)
    _, markers = cv2.connectedComponents(sure_foreground)

    markers += 1
    markers[unknown == 255] = 0
    markers = markers.astype('int32')
    markers = cv2.watershed(returnImg, markers)

    labels = np.unique(markers)
    objects = []
    for lbl in labels[2:]:
        target = np.where(markers == lbl, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects.append(contours[0])

    returnImg = cv2.drawContours(returnImg, objects, -1, color=(0, 0, 255), thickness=2)
    return returnImg

class TransformPipeline:
    def __init__(self, transformations: List[Callable[[np.ndarray], np.ndarray]]):
        """
        Initialize the pipeline with a list of transformations.

        Parameters:
        - transformations: List[Callable[[np.ndarray], np.ndarray]] : A list of functions
          that take a numpy array as input and return a numpy array.
        """
        self.transformations = transformations

    def __call__(self, img: Union[np.ndarray, str]) -> np.ndarray:
        """Make the object callable."""
        return self.apply(img)

    def apply(self, img: Union[np.ndarray, str]) -> np.ndarray:
        """
        Apply all transformations to the given image.

        Parameters:
        - img: Union[np.ndarray, str] : The image to transform, can be a numpy array or a file path.

        Returns:
        - np.ndarray : The transformed image.
        """
        if isinstance(img, str):
            img = img_to_numpy_array(img)
        for transform in self.transformations:
            img = transform(img)
        return img

def convert_to_rgb(img):
    if isinstance(img, np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
