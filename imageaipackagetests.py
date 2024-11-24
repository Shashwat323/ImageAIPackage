import os
import unittest
import imageaipackage as iap


class ImageAIPackageTests(unittest.TestCase):
    def test_resize(self):
        new_img = iap.resize(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), 250)
        iap.save_image(new_img, "unit_test_images/ResizedImg1.jpg")

    def test_crop(self):
        new_img = iap.crop(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), 150)
        iap.save_image(new_img, "unit_test_images/CroppedImg1.jpg")

    def test_region_grow(self):
        iap.region_grow(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), 10)

if __name__ == '__main__':
    unittest.main()
