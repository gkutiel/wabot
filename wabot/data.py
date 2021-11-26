from torch.utils.data import DataLoader
from datetime import timedelta
from typing import Iterable

from torch.utils.data.dataset import Dataset

from wabot.parser import Msg


class MsgDataset(Dataset):
    def __init__(self, msgs: Iterable[Msg]):
        self.msgs = list(msgs)

    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, i):
        return self.msgs[i]


def msg_data_loader(msgs: Iterable[Msg]) -> DataLoader:
    return DataLoader(
        MsgDataset(msgs),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x)
