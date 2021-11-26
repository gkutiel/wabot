from dataclasses import dataclass
from torch.utils.data import DataLoader
from datetime import timedelta
from typing import Iterable, List

from torch.utils.data.dataset import Dataset

from wabot.parser import Msg


@dataclass(frozen=True)
class Data:
    sender_id: int
    tokens: List[int]


class SenderDataset(Dataset):
    def __init__(self, msgs: Iterable[Msg]):
        self.msgs = list(msgs)

    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, i):
        return self.msgs[i]


def msg_data_loader(msgs: Iterable[Msg], shuffle=False) -> DataLoader:
    return DataLoader(
        SenderDataset(msgs),
        batch_size=1,
        shuffle=shuffle,
        collate_fn=lambda x: x)
