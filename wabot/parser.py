import re

from datetime import datetime
from dataclasses import dataclass

datetime_msg_re = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{1,2}) - (.*)$')


@dataclass
class Line:
    time: datetime
    msg: str


def parse_line(line: str):
    match = datetime_msg_re.match(line)
    if not match:
        raise ValueError(f'Invalid line: {line}')

    return Line(
        time=datetime.strptime(match[1], '%m/%d/%y, %H:%M'),
        msg=match[2])
