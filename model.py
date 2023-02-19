import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms

if __name__ == '__main__':
    dataset = ImageFolder(".\\train", transform=transforms.Compose([
        transforms.CenterCrop((200, 800)),
        ToTensor()
    ]))

    img, label = dataset[0]
    print(img.shape, label)
    print(len(dataset.classes))

    batch_size = 64
    random_seed = 8
    torch.manual_seed(random_seed)
    val_size = 1000
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_ds), len(val_ds))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)


    class ImageClassificationBase(nn.Module):
        def training_step(self, inputs, labels):
            out = self(inputs)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))


    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    class Cifar10CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 400 100
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 200 50
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 100 25
                nn.Flatten(),
                nn.Linear(64 * 100 * 25, 320)
            )

        def forward(self, xb):
            return self.network(xb)


    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


    device = get_default_device()
    print(device)


    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            print(epoch+1)
            # Training Phase
            model.train()
            train_losses = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = model.training_step(inputs, labels)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                result = evaluate(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                model.epoch_end(epoch, result)
                history.append(result)
                torch.cuda.empty_cache()
        return history


    model = Cifar10CnnModel()
    model.to(device)
    # for images, labels in train_dl:
    #     print('images.shape:', images.shape)
    #     images, labels = images.to(device), labels.to(device)
    #     out = model(images)
    #     print('out.shape:', out.shape)
    #     print('out[0]:', out[0])
    #     break

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = math.exp(-4)

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print(history)
