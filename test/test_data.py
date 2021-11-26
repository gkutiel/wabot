from datetime import datetime
from wabot.data import Data
from wabot.parser import Msg


def test_MsgDataset():
    from wabot.data import SenderDataset
    data = [Data(i, '') for i in range(10)]
    ds = SenderDataset(data)

    assert len(ds) == 10
    assert ds[0] == data[0]


def test_msg_data_loader():
    from wabot.data import msg_data_loader
    senders = {'a': 0, 'b': 1, 'c': 2}
    msgs = [Msg(datetime(2020, 1, i+1), 'a', 'Hello World') for i in range(10)]
    dl = msg_data_loader(msgs, senders)

    assert len(dl) == 10
