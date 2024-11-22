import os
import unittest
import imageaipackage
from imageaipackage import image_path_to_np_array, resize


class ImageAIPackageTests(unittest.TestCase):
    def test_resize(self):
        new_img = resize(image_path_to_np_array("unit_test_images/TestImg1.jpg"))
        imageaipackage.save_image(new_img, "unit_test_images/ResizedImg1.jpg")

    def test_crop(self):
        new_img = resize(image_path_to_np_array("unit_test_images/TestImg1.jpg"), True)
        imageaipackage.save_image(new_img, "unit_test_images/CroppedImg1.jpg")


if __name__ == '__main__':
    unittest.main()
