from dataclasses import dataclass
from torch.utils.data import DataLoader
from datetime import timedelta
from typing import Dict, Iterable, List

from torch.utils.data.dataset import Dataset

from wabot.parser import Msg, Tokenizer


@dataclass(frozen=True)
class Data:
    sender_id: int
    text: str


class SenderDataset(Dataset):
    def __init__(self, data: List[Data]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def msg_data_loader(msgs: Iterable[Msg], senders: Dict[str, int], shuffle=False) -> DataLoader:
    data = [Data(senders[msg.sender], msg.text) for msg in msgs]
    return DataLoader(
        SenderDataset(data),
        batch_size=1,
        shuffle=shuffle,
        collate_fn=lambda x: x)
