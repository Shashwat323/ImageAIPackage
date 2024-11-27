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

    def test_gray_scale(self):
        new_img = imageaipackage.gray_scale("unit_test_images/TestImg1.jpg", 1)
        imageaipackage.save_image(new_img, "unit_test_images/grayScaledImage.jpg")

    def test_random_crop(self):
        new_img = imageaipackage.random_crop("unit_test_images/TestImg1.jpg")
        imageaipackage.save_image(new_img, "unit_test_images/random_cropped_image.jpg")

    def test_region_grow(self):
        new_img = iap.region_grow(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 10)
        iap.save_image(new_img, "unit_test_images/RegionGrowImg1.jpg")

    def test_overlay_images(self):
        base_img = iap.image_path_to_np_array("unit_test_images/TestImg1.jpg")
        overlay_img = iap.image_path_to_np_array("unit_test_images/TestImg2.jpg")
        new_img = iap.overlay_images(base_img, overlay_img, transparency=0.5)
        iap.save_image(new_img, "unit_test_images/overlay_images_test.jpg")

    def test_adjust_contrast(self):
        new_img = iap.adjust_contrast(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), factor=1.5, chance=1.0)
        iap.save_image(new_img, "unit_test_images/adjust_contrast_test.jpg")

    def test_adjust_hue(self):
        new_img = iap.adjust_hue(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), factor=1.2, chance=1.0)
        iap.save_image(new_img, "unit_test_images/adjust_hue_test.jpg")

    def test_adjust_brightness(self):
        new_img = iap.adjust_brightness(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), factor=1.2, chance=1.0)
        iap.save_image(new_img, "unit_test_images/adjust_brightness_test.jpg")

    def test_square_rotate(self):
        new_img = iap.square_rotate(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), mode=2, chance=1.0)
        iap.save_image(new_img, "unit_test_images/square_rotate_test.jpg")

    def test_mirror_image(self):
        new_img = iap.mirror_image(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"), chance=1.0)
        iap.save_image(new_img, "unit_test_images/mirror_image_test.jpg")

    def test_random_rotate(self):
        new_img = iap.random_rotate(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"))
        iap.save_image(new_img, "unit_test_images/random_rotate_test.jpg")



if __name__ == '__main__':
    unittest.main()
