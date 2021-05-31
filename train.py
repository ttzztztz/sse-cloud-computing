import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


batch_size, learning_rate = 64, 0.02
num_epoches = 20

data_tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
model = CNN()

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
epoch = 0

for data in train_loader:
    img, label = data
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

torch.save(model, 'mnist.pth')

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

print('Loss: {:.6f}%, Acc: {:.6f}%'.format(
    100.0 * eval_loss / (len(test_dataset)),
    100.0 * eval_acc / (len(test_dataset))
))
