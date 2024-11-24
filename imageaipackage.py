import itertools
import random

import cv2
import numpy as np
from PIL import Image


def gray_scale(image_file_path: str, mode: int = 1):
    # mode 1 = gray scale 0-255
    # mode 2 = 8 bit gray scale
    # mode 3 = black and white
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
    # return image here. (Return model to be decided)


def save_image(img: np.ndarray, output_file_path: str) -> None:
    if img is None:
        raise ValueError("Cannot save an image that is None.")

    # Save the image
    cv2.imwrite(output_file_path, img)

def image_path_to_np_array(image_file_path: str) -> np.ndarray:
    img = cv2.imread(image_file_path)
    return img

def crop(img: np.ndarray, new_width: int = float('inf')) -> np.ndarray:
    """Crop to given width or images lowest height/width, whichever is smaller"""
    height, width = img.shape[:2]
    size = min(height, width, new_width)
    x = int((width - size) / 2)
    y = int((height - size) / 2)
    return img[y:y + size, x:x + size]

def resize(img: np.ndarray, new_width: int = 224) -> np.ndarray:
    return cv2.resize(img, (new_width, new_width))

def region_grow(img: np.ndarray, threshold: int = 10):
    regiongrow = RegionGrow(img, threshold)
    regiongrow.apply_region_grow()

class Stack:
    def __init__(self):
        self.item = []
        self.obj = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []


class RegionGrow:

    def __init__(self, img, th):
        self.read_image(img)
        self.h, self.w, _ = self.im.shape
        self.passedBy = np.zeros((self.h, self.w), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def read_image(self, img):
        self.im = img.astype('int')

    def get_neighbour(self, x0, y0):
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.boundaries(x := x0 + i, y := y0 + j)
        ]

    def create_seeds(self):
        return [
            [self.h / 2, self.w / 2],
            [self.h / 3, self.w / 3], [2 * self.h / 3, self.w / 3], [self.h / 3 - 10, self.w / 3],
            [self.h / 3, 2 * self.w / 3], [2 * self.h / 3, 2 * self.w / 3], [self.h / 3 - 10, 2 * self.w / 3],
            [self.h / 3, self.w - 10], [2 * self.h / 3, self.w - 10], [self.h / 3 - 10, self.w - 10]
        ]

    def apply_region_grow(self, cv_display=True):

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

        if cv_display:
            [self.color_pixel(i, j) for i, j in itertools.product(range(self.h), range(self.w))]
            self.display()

    def reset_region(self, x0, y0):

        self.passedBy[self.passedBy == self.currentRegion] = 0
        x0 = random.randint(x0 - 4, x0 + 4)
        y0 = random.randint(y0 - 4, y0 + 4)
        x0 = np.clip(x0, 0, self.h - 1)
        y0 = np.clip(y0, 0, self.w - 1)
        self.currentRegion -= 1
        return x0, y0

    def color_pixel(self, i, j):
        val = self.passedBy[i][j]
        self.SEGS[i][j] = (255, 255, 255) if (val == 0) else (val * 35, val * 90, val * 30)

    def display(self):
        cv2.imshow("", self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bfs(self, x0, y0):

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

        return self.iterations > max_iteration or np.all(self.passedBy > 0)

    def boundaries(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def distance(self, x, y, x0, y0):

        return np.linalg.norm(self.im[x0, y0] - self.im[x, y])