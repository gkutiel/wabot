import pytest

from datetime import datetime


def test_date_time_msg_re():
    from wabot.parser import datetime_msg_re

    assert not datetime_msg_re.match('Dec 12, 00:00 - Hello World!')

    match = datetime_msg_re.match('01/01/18, 00:00 - Hello World!')
    assert match
    assert match[1] == '01/01/18, 00:00'
    assert match[2] == 'Hello World!'

    match = datetime_msg_re.match('1/1/18, 0:00 - Hello World!')
    assert match
    assert match[1] == '1/1/18, 0:00'
    assert match[2] == 'Hello World!'


def test_parse_line():
    from wabot.parser import parse_line

    line = parse_line('01/01/18, 00:00 - Hello World!')
    assert line.time == datetime(2018, 1, 1, 0, 0)
    assert line.msg == 'Hello World!'

    line = parse_line('1/1/18, 0:00 - Hello World!')
    assert line.time == datetime(2018, 1, 1, 0, 0)
    assert line.msg == 'Hello World!'

    with pytest.raises(ValueError):
        parse_line('Dec 12, 00:00 - Hello World!')
