import os
import unittest
import imageaipackage

class ImageAIPackageTests(unittest.TestCase):
    def test_resize(self):
        new_img = imageaipackage.resize("unit_test_images/TestImg1.jpg")
        imageaipackage.save_image(new_img, "unit_test_images/ResizedImg1.jpg")

    def test_crop(self):
        new_img = imageaipackage.resize("unit_test_images/TestImg1.jpg", True)
        imageaipackage.save_image(new_img, "unit_test_images/CroppedImg1.jpg")

    def test_gray_scale(self):
        new_img = imageaipackage.gray_scale("unit_test_images/TestImg1.jpg", 1)
        imageaipackage.save_image(new_img, "unit_test_images/grayScaledImage.jpg")

    def test_random_crop(self):
        new_img = imageaipackage.random_crop("unit_test_images/TestImg1.jpg")
        imageaipackage.save_image(new_img, "unit_test_images/random_cropped_image.jpg")

if __name__ == '__main__':
    unittest.main()
