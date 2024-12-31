from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch
import cv2

from models.models import get_model
import loader

def segment_test_and_show(img_dir, model_weights_path):
    #return model_weights_path
    model_onnx = YOLO(model_weights_path)
    img = cv2.imread(img_dir)
    results = model_onnx(img)
    results[0].show()
    return results

def test_and_show(img_dir, weight_dir, to_tensor, model="default", label_transform=None):
    device = "cpu"
    image = Image.open(img_dir)
    # open and transform image for vit
    image_vit = to_tensor(img_dir)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    model = get_model(model_type=model)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device, weights_only=True))

    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    #print(pred)
    pred_label = torch.argmax(pred)
    if label_transform:
        pred_label = label_transform(pred_label)

    # plot
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted Label: {pred_label} and pred {pred}")
    plt.show()
    return pred_label


if __name__ == "__main__":
    #test_and_show("unit_test_images/one.png", 'D:\\Other\\Repos\\ImageAIPackage\\weights\\20241205_131821_number.pt',
                  #model="number_simple_cnn", to_tensor=loader.number_tensor, label_transform=None)
    segment_test_and_show("unit_test_images/TREE4.jpg", "C:\\Users\\cohen\\OneDrive\\Documents\\ImageAIPackage\\weights\\branches.pt")