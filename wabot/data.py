import torch
from torch.utils.data import DataLoader
from typing import Dict, Iterable

from torch.utils.data.dataset import TensorDataset
from wabot.TextEncoder import TextEncoder

from wabot.parser import Msg
from torch.nn.utils.rnn import pad_sequence


def sender_dataloader(
        *, msgs: Iterable[Msg],
        senders: Dict[str, int],
        text_encoder: TextEncoder,
        min_tokens=5,
        batch_size=1024,
        shuffle=False) -> DataLoader:

    data = [(senders[msg.sender], text_encoder.tokenize(msg.text)) for msg in msgs]
    data = [(sid, tokens) for sid, tokens in data if len(tokens) >= min_tokens]

    print(f'Len data {len(data)}')

    sids, tokens = zip(*data)

    sids = torch.tensor(sids)

    tokens = pad_sequence(
        [torch.tensor(t) for t in tokens],
        batch_first=True,
        padding_value=0)

    return DataLoader(
        TensorDataset(sids, tokens),
        batch_size=batch_size,
        shuffle=shuffle)
