from datetime import datetime
from wabot.parser import Msg


def test_to_sessions():
    from wabot.data import to_sessions

    def msg(h, m):
        return Msg(
            time=datetime(2018, 1, 1, h, m),
            sender='Bob',
            text='')

    msgs = [msg(0, 0), msg(0, 1), msg(0, 2), msg(0, 3), msg(0, 4), msg(0, 5)]
    assert len(list(to_sessions(msgs))) == 1

    msgs = [msg(0, 0), msg(0, 1), msg(0, 2), msg(0, 20), msg(0, 21), msg(0, 50)]
    sessions = list(to_sessions(msgs))
    assert len(sessions[0]) == 3
    assert len(sessions[1]) == 2
    assert len(sessions[2]) == 1
