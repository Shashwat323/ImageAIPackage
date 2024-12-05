import numpy as np
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

    def test_random_crop(self):
        new_img = iap.random_crop(iap.image_path_to_np_array("unit_test_images/TestImg1.jpg"))
        iap.save_image(new_img, "unit_test_images/random_cropped_image.jpg")

    def test_region_grow(self):
        new_img = iap.region_grow(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 10)
        iap.save_image(new_img, "unit_test_images/RegionGrowImg1.jpg")

    def test_sharpen(self):
        new_img = iap.sharpen(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 9)
        iap.save_image(new_img, "unit_test_images/SharpenImg1.jpg")

    def test_watershed(self):
        new_img = iap.watershed(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"))
        iap.save_image(new_img, "unit_test_images/WatershedImg1.jpg")

    def test_edge_detection(self):
        new_img = iap.edge_detection(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"))
        iap.save_image(new_img, "unit_test_images/edgeDetectionImg1.jpg")

    def test_otsus_binarization_thresholding(self):
        new_img = iap.otsus_binarization_thresholding(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"))
        iap.save_image(new_img, "unit_test_images/edgeDetectionImg1.jpg")

    def test_adaptive_thresholding(self):
        new_img = iap.adaptive_thresholding(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"))
        iap.save_image(new_img, "unit_test_images/adaptiveThresholdingImg1.jpg")

    def test_custom_kernel_blur(self):
        new_img = iap.custom_kernel_blur(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        iap.save_image(new_img, "unit_test_images/customKernelBlurImg1.jpg")

    def test_laplacian(self):
        new_img = iap.laplacian(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 3)
        iap.save_image(new_img, "unit_test_images/laplacianImg1.jpg")

    def test_median_blur(self):
        new_img = iap.median_blur(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 3)
        iap.save_image(new_img, "unit_test_images/medianBlurImg1.jpg")

    def test_gaussian_blur(self):
        new_img = iap.gaussian_blur(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 3)
        iap.save_image(new_img, "unit_test_images/gaussianBlurImg1.jpg")

    def test_bilateral_blur(self):
        new_img = iap.bilateral_blur(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 3)
        iap.save_image(new_img, "unit_test_images/bilateralBlurImg1.jpg")

    def test_box_blur(self):
        new_img = iap.blur(iap.image_path_to_np_array("unit_test_images/TestImg2.jpg"), 3)
        iap.save_image(new_img, "unit_test_images/blurImg1.jpg")



    def test_overlay_images(self):
        base_img = iap.image_path_to_np_array("unit_test_images/TestImg1.jpg")
        overlay_img = iap.image_path_to_np_array("unit_test_images/TestImg1.jpg")
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
