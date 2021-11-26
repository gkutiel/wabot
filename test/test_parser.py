import pytest

from datetime import datetime

from wabot.parser import Msg, get_senders


def test_date_time_msg_re():
    from wabot.parser import datetime_msg_re

    assert not datetime_msg_re.match('Dec 12, 00:00 - Bob: Hello World!')

    match = datetime_msg_re.match('01/01/18, 00:00 - Bob: Hello World!')
    assert match
    assert match[1] == '01/01/18, 00:00'
    assert match[2] == 'Bob'
    assert match[3] == 'Hello World!'

    match = datetime_msg_re.match('1/1/18, 0:00 - Bob: Hello World!')
    assert match
    assert match[1] == '1/1/18, 0:00'
    assert match[2] == 'Bob'
    assert match[3] == 'Hello World!'


def test_parse_line():
    from wabot.parser import parse_msg

    line = parse_msg('01/01/18, 00:00 - Bob: Hello World!')
    assert line.time == datetime(2018, 1, 1, 0, 0)
    assert line.sender == 'Bob'
    assert line.text == 'Hello World!'

    line = parse_msg('1/1/18, 0:00 - Bob: Hello World!')
    assert line.time == datetime(2018, 1, 1, 0, 0)
    assert line.sender == 'Bob'
    assert line.text == 'Hello World!'

    with pytest.raises(ValueError):
        parse_msg('Dec 12, 00:00 - Bob: Hello World!')


def test_parse():
    from wabot.parser import parse

    lines = [
        '01/01/18, 00:00 - Bob: Hello World!',
        '01/01/18, 00:01 - Bob: Hello World!',
        '01/01/18, 00:02 - Bob: Hello World!',
    ]

    parsed = list(parse(lines))
    assert len(parsed) == 3
    assert parsed[0].time == datetime(2018, 1, 1, 0, 0)
    assert parsed[0].text == 'Hello World!'
    assert parsed[1].time == datetime(2018, 1, 1, 0, 1)
    assert parsed[1].text == 'Hello World!'
    assert parsed[2].time == datetime(2018, 1, 1, 0, 2)
    assert parsed[2].text == 'Hello World!'


def msg(h=0, m=0, s='Bob', t='Hi'):
    return Msg(
        time=datetime(2018, 1, 1, h, m),
        sender=s,
        text=t)


def test_get_senders():
    assert get_senders([msg(), msg(), msg()]) == ['Bob']
    assert get_senders([msg(s='A'), msg(s='C'), msg(s='B')]) == ['A', 'B', 'C']


def test_to_sessions():
    from wabot.parser import get_sessions

    msgs = [msg(0, 0), msg(0, 1), msg(0, 2), msg(0, 3), msg(0, 4), msg(0, 5)]
    assert len(list(get_sessions(msgs))) == 1

    msgs = [msg(0, 0), msg(0, 1), msg(0, 2), msg(0, 20), msg(0, 21), msg(0, 50)]
    sessions = list(get_sessions(msgs))
    assert len(sessions[0]) == 3
    assert len(sessions[1]) == 2
    assert len(sessions[2]) == 1
