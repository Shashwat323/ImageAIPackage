curl -X GET -F "model_name=resnet50_cifar10" -F "model_weights=@C:\\Users\\cohen\\OneDrive\\Documents\\ImageAIPackage\\api-uploads\\cifar10_resnet50.pt"  -F "image_id=b4469f58-a9f7-44d4-922f-8938a6f5c050" http://127.0.0.1:5000/predict-image
echo "Image prediction complete. Press any key to exit..."
read -n 1 -s