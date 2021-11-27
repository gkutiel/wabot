import re
import json

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Iterable, List, cast
from collections import Counter

from wabot.Params import Params

datetime_msg_re = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{1,2}) - ([^:]{1,20}): (.*)$')


@dataclass(frozen=True)
class Msg:
    time: datetime
    sender: str
    text: str

    def __str__(self):
        return f'{self.time} - {self.sender}: {self.text}'


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

    return sorted(tokens)


def build_lexicon(tokens: List[str], params: Params, save=False) -> Dict[str, int]:
    words, _ = zip(*Counter(tokens).most_common(params.lexicon_size))
    words = cast(List[str], words)
    lexicon = dict((w, i+1) for i, w in enumerate(words))

    if save:
        json.dump(
            lexicon,
            open('lexicon.json', 'w', encoding='utf-8'),
            ensure_ascii=False)

    return lexicon


def get_senders(msgs: Iterable[Msg], save=False):
    senders = sorted(set(msg.sender for msg in msgs))

    if save:
        json.dump(
            senders,
            open('senders.json', 'w', encoding='utf-8'),
            ensure_ascii=False)

    return senders


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


def get_sessions(msgs: Iterable[Msg], max_gap=timedelta(minutes=2)):
    session = []
    for msg in msgs:
        if not session or (msg.time - session[-1].time) < max_gap:
            session.append(msg)
        else:
            yield session
            session = [msg]

    yield session
