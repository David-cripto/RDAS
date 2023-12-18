from pathlib import Path
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset

PathLike = Path | str

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class ClassMnist(Dataset):
    def __init__(self, img_dir, train=True, label = 0):
        super().__init__()
        self.train_dataset = torchvision.datasets.MNIST(
            img_dir, download=True, train=train
        )
        self.label = label
        self.dataset = self.zero_filter()

    def zero_filter(self):
        dataset = []
        for i in range(len(self.train_dataset)):
            if self.train_dataset[i][1] == self.label:
                dataset.append((TRANSFORM(self.train_dataset[i][0]), self.train_dataset[i][1]))
        return dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.dataset[i]


def get_dataset(*args, **kwargs) -> tuple[Dataset, ...]:
    return ClassMnist(*args, **kwargs)