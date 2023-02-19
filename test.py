import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms

if __name__ == '__main__':
    dataset = ImageFolder(".\\train", transform=transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Grayscale(),
        ToTensor()
    ]))

    img, label = dataset[0]
    print(img.shape, label)
    print(dataset.classes)

    batch_size = 64
    random_seed = 8
    torch.manual_seed(random_seed)
    val_size = 800
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_ds), len(val_ds))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, 2 * batch_size, num_workers=4, pin_memory=True)


    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    class CnnModel(nn.Module):
        def __init__(self):
            super(CnnModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(12)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(24)
            self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(24)
            self.fl = nn.Flatten()
            self.fc1 = nn.LazyLinear(320)

        def forward(self, input):
            output = F.relu(self.bn1(self.conv1(input)))
            output = F.relu(self.bn2(self.conv2(output)))
            output = self.pool(output)
            output = F.relu(self.bn4(self.conv4(output)))
            output = F.relu(self.bn5(self.conv5(output)))
            output = self.fl(output)
            output = self.fc1(output)
            return output


    model = CnnModel()


    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


    def to_device(data, device):

        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


    class DeviceDataLoader:
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            return len(self.dl)


    device = get_default_device()


    def evaluate(model, val_loader):
        total_acc = 0
        len_preds = 0
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            out = model.forward(inputs)
            _, preds = torch.max(out, dim=1)
            print("acc in batch of val data")
            f1 = torchmetrics.F1Score(task="multiclass", num_classes=320)
            f1.cuda()
            print(f1(labels, preds))
            total_acc += torch.tensor(torch.sum(preds == labels).item())
            len_preds += len(preds)
        return total_acc / len_preds


    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        result = torch.tensor(0.)
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = 0
            for inputs, labels in train_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = F.cross_entropy(outputs, labels)
                train_losses += loss.item()
                loss.backward()
                optimizer.step()
                # Validation phase
            print("End of epoch:", epoch, " running loss: ", train_losses)
            with torch.no_grad():
                model.eval()
                result1 = evaluate(model, val_loader)
                if result1 > result:
                    torch.save(model.state_dict(), 'CNN_Model.pth')
                    print(result1)
        return result


    model = model.cuda()
    # for images, labels in train_dl:
    #     print('images.shape:', images.shape)
    #     images, labels = images.to(device), labels.to(device)
    #     out = model(images)
    #     print('out.shape:', out.shape)
    #     print('out[0]:', out[0])
    #     break

    num_epochs = 20
    opt_func = torch.optim.SGD
    lr = 0.001

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
