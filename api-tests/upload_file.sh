curl -X POST -F "file=@C:\\Users\\cohen\\OneDrive\\Documents\\ImageAIPackage\\weights\\cifar10_resnet50.pt" http://127.0.0.1:5000/upload-file
echo "File upload complete. Press any key to exit..."
read -n 1 -s