from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(self, items):
        super(DummyDataset, self).__init__()
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    a = list(range(1000))
    dataset = DummyDataset(a)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batches = [x for x in loader]
~