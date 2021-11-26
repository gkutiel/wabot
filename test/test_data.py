from datetime import datetime
from wabot.parser import Msg


def test_MsgDataset():
    from wabot.data import SenderDataset
    msgs = [Msg(datetime(2020, 1, i+1), 'sender', ['שלום', 'עולם']) for i in range(10)]
    ds = SenderDataset(msgs)

    assert len(ds) == 10
    assert ds[0] == msgs[0]


def test_msg_data_loader():
    from wabot.data import msg_data_loader
    msgs = [Msg(datetime(2020, 1, i+1), 'sender', ['שלום', 'עולם']) for i in range(10)]
    dl = msg_data_loader(msgs)

    assert len(dl) == 10
