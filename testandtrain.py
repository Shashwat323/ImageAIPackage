'''Train CIFAR10 with PyTorch.'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim

import run
from models import resnet
from models.models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = resnet.ResNet152(3, 10)
#net.load_state_dict(torch.load('ckpt.pth', weights_only=False, map_location='cpu')['net'])
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

"""def test_image(img):
    img = iap.img_to_numpy_array(img)
    img = iap.resize(img, 32)
    img = func.to_tensor(img)

    net.eval()
    with torch.no_grad():
        result = net.forward(img.unsqueeze(0))
        _, prediction = result.max(1)
        print('Model Predicts the image is:', classes[prediction])
        img = img.permute(1, 2, 0)
        imgnp = img.numpy()
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions                 # Read image
        img = cv2.resize(imgnp, (64, 64))                # Resize image
        cv2.imshow("output", img)                       # Show image
        cv2.waitKey(0)"""


if __name__ == '__main__':
    #test_image('unit_test_images/CAT1.jpg')
    #test(0)

    for epoch in range(start_epoch, start_epoch+1):
        run.train(train_loader=trainloader, model=net, loss_fn=criterion, optimizer=scheduler, use_progress_bar=True)
        run.test(test_loader=testloader, model=net, loss_fn=criterion, use_progress_bar=True)
        scheduler.step()