from produce_dataset import DatasetRegularProcess


class InitDataset:
    def __init__(self):
        self.operate = DatasetRegularProcess()

    def __getitem__(self, item):
