curl -X GET -F "image_id=be8ec72d-6fdc-462e-b3b0-d3f67ffe6123" -F "model_id=818c1e5e-3678-4dec-9e32-d89d96a2f68c" http://127.0.0.1:5000/create-mask --output processed.jpg
echo "Image prediction complete. Press any key to exit..."
read -n 1 -s