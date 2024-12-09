from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor

from models import get_model
import loader


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

    print(pred)
    pred_label = torch.argmax(pred)
    if label_transform:
        pred_label = label_transform(pred_label)

    # plot
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted Label: {pred_label}")
    plt.show()


if __name__ == "__main__":
    test_and_show("unit_test_images/horse.jpg", 'D:\\Other\\Repos\\ImageAIPackage\\weights\\20241205_135518_cifar10.pt',
                  model="cifar10_simple_cnn", to_tensor=loader.tensor, label_transform=loader.cifar_index_to_label)