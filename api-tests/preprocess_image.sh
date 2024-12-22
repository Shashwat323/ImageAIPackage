curl -X POST -F "image_id=a977638f-8356-4d55-9c87-b1a2e8bb8dc2" -F "technique=random_crop" http://127.0.0.1:5000/preprocess-image --output processed.jpg
echo "Image preprocess complete. Press any key to exit..."
read -n 1 -s