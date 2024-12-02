from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import tensor, normalize, flower_index_to_label


def test_and_show(img_dir, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = normalize(image)
    image_vit = tensor(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    print(pred)
    pred_label = torch.argmax(pred)
    pred_label = flower_index_to_label(pred_label)

    # plot
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted Label: {pred_label}")
    plt.show()


if __name__ == "__main__":
    pred = test_and_show('unit_test_images/sunflower.jpg', 'weights/20241202_012102.pt')