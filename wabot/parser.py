import re

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Iterable

datetime_msg_re = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{1,2}) - ([^:]{1,20}): (.*)$')


@dataclass
class Msg:
    time: datetime
    sender: str
    text: str

    def to_dict(self):
        return {
            'time': self.time.isoformat(),
            'sender': self.sender,
            'text': self.text}


def parse_msg(line: str):
    match = datetime_msg_re.match(line)
    if not match:
        raise ValueError(f'Invalid line: {line}')

    return Msg(
        time=datetime.strptime(match[1], '%m/%d/%y, %H:%M'),
        sender=match[2],
        text=match[3])


def parse(lines: Iterable[str]):
    for line in lines:
        try:
            yield parse_msg(line)
        except ValueError as e:
            pass


def get_senders(msgs: Iterable[Msg]):
<<<<<<< HEAD
    senders = set()
    for msg in msgs:
        senders.add(msg.sender)
    return sorted(list(senders))


def to_sessions(msgs: Iterable[Msg], max_gap_minutes=10):
=======
    return sorted(list(set(msg.sender for msg in msgs)))


def get_sessions(msgs: Iterable[Msg], max_gap_minutes=10):
>>>>>>> f
    max_td = timedelta(minutes=max_gap_minutes)
    session = []
    for msg in msgs:
        if not session or msg.time - session[-1].time < max_td:
            session.append(msg)
        else:
            yield session
            session = [msg]

    yield session
