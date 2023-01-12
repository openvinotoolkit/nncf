import torch
from torch import nn
from torch.optim import SGD
from torch.utils import data
from torch.utils.data import DataLoader


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            n_classes=2,
            depth=1,
            wf=1,
            padding=True
    ):
        super().__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        for i, down in enumerate(self.down_path):
            x = down(x)
        x = self.last(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super().__init__()
        self.block = nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding))

    def forward(self, x):
        out = self.block(x)
        return out


class MockDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        image = torch.rand((3, 23, 30))
        target = torch.randint(0, 1, (23, 30))
        return image, target


def main():
    train_set = MockDataset()
    train_loader = DataLoader(train_set, batch_size=4, num_workers=1, drop_last=True)
    model = UNet(n_classes=13)
    device = 'cpu'
    model.to(device)
    print(model)
    optimizer = SGD(model.parameters(), lr=1e-3)
    for epoch in range(1):
        model.train()
        for step, batch_data in enumerate(train_loader):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape, labels.shape)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            print(f"\nBefore HANG {loss}\n")
            loss.backward()
            print("\nAFTER HANG\n")


if __name__ == '__main__':
    import os

    for k, v in os.environ.items():
        print(f'{k}={v}')
    main()
