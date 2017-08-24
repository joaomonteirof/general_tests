from torch.utils.data import Dataset

class DummyMultiDataset(Dataset):

    def __init__(self, a, b, c):
        super(DummyMultiDataset, self).__init__()
        assert len(a) == len(b) == len(c)
        self.a = a
        self.b = b
        self.c = c

    def __getitem__(self, index):
        return self.a[index], self.b[index], self.c[index]

    def __len__(self):
        return len(self.a)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy, torch

    a = torch.FloatTensor(numpy.random.randn(1000, 10))
    b = torch.LongTensor(numpy.random.randint(10, size=(1000,)))
    c = list(range(1000))

    dataset = DummyMultiDataset(a, b, c)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batches = [(x, y, z) for x, y, z in loader]