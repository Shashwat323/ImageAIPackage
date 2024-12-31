curl -X GET -F "model_weights=C:\\Users\\cohen\\OneDrive\\Documents\\ImageAIPackage\\weights\\branches.pt"  -F "image_id=16c7887f-3050-4b6a-afd6-148aec29a341" http://127.0.0.1:5000/segment-image
echo "Image segmentation complete. Press any key to exit..."
read -n 1 -s