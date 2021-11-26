import torch
from torch.utils.data import DataLoader
from typing import Dict, Iterable

from torch.utils.data.dataset import TensorDataset
from wabot.TextEncoder import TextEncoder

from wabot.parser import Msg
from torch.nn.utils.rnn import pad_sequence

from wabot.train import Params


def sender_dataloader(
        *, msgs: Iterable[Msg],
        senders: Dict[str, int],
        text_encoder: TextEncoder,
        params: Params) -> DataLoader:

    data = [(senders[msg.sender], text_encoder.tokenize(msg.text)) for msg in msgs]
    data = [(sid, tokens) for sid, tokens in data if len(tokens) >= params.min_tokens]

    sids, tokens = zip(*data)

    sids = torch.tensor(sids)

    tokens = pad_sequence(
        [torch.tensor(t) for t in tokens],
        batch_first=True,
        padding_value=0)

    return DataLoader(
        TensorDataset(sids, tokens),
        batch_size=params.batch_size)
