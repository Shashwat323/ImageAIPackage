curl -X POST -F "image=@D:\\Other\\Repos\\ImageAIPackage\\unit_test_images\\daisy-drawing.jpg" -F "technique=random_crop" http://127.0.0.1:5000/upload-image --output processed_image.jpg
echo "Processing complete. Press any key to exit..."
read -n 1 -s