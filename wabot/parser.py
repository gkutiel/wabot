import re

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Iterable, List
from collections import Counter

datetime_msg_re = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{1,2}) - ([^:]{1,20}): (.*)$')


@dataclass(frozen=True)
class Msg:
    time: datetime
    sender: str
    text: str


class Tokenizer:
    def __init__(
            self, *,
            sub=re.compile(r'[^אבגדהוזחטיכךלמםנןסעפףצץקרשת]+'),
            split=re.compile(r'\s+')):
        self.sub = sub
        self.split = split

    def __call__(self, text):
        return [t for t in self.split.split(self.sub.sub(' ', text)) if t]


def get_messages(chat_txt='chat.txt'):
    lines = open(chat_txt).readlines()
    return list(parse(lines))


def get_tokens(msgs: Iterable[Msg], tokenizer=Tokenizer()):
    tokens = set()
    for msg in msgs:
        tokens.update(tokenizer(msg.text))

    return tokens


def build_lexicon(tokens: Iterable[str], size=1000):
    words, _ = zip(*Counter(tokens).most_common(size))
    return dict((w, i) for i, w in enumerate(words))


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
    return dict((u, i) for i, u in enumerate(set(msg.sender for msg in msgs)))


def get_sessions(msgs: Iterable[Msg], max_gap=timedelta(minutes=10)):
    session = []
    for msg in msgs:
        if not session or (msg.time - session[-1].time) < max_gap:
            session.append(msg)
        else:
            yield session
            session = [msg]

    yield session
