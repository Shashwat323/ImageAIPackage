curl -X GET -F "model_name=resnet50" -F "model_id=eb672766-c8d1-4d69-9304-73a4c4c1f4d9"  -F "image_id=b4469f58-a9f7-44d4-922f-8938a6f5c050" http://127.0.0.1:5000/predict-image
echo "Image prediction complete. Press any key to exit..."
read -n 1 -s