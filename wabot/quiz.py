from datetime import timedelta
import json

from tqdm import tqdm
from typing import List, cast
from torch import Tensor
from dataclasses import dataclass
from wabot.Params import Params
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.parser import Msg, get_messages, get_sessions


def window(
        msgs: List[Msg],
        pos: int,
        delta: timedelta = timedelta(minutes=180),
        max_size: int = 5):

    time = msgs[pos].time

    return [
        msg
        for msg
        in msgs[pos-max_size:pos+max_size]
        if max(msg.time, time) - min(msg.time, time) < delta
        and not '<Media omitted>' in msg.text]


def make_questions(msgs, params=Params()):
    model = SimpleSenderPredictor.load_from_checkpoint('model.ckpt')
    for pos, msg in tqdm(enumerate(msgs)):
        tokens = model.text_encoder.tokenize(msg.text)
        if params.min_tokens <= len(tokens) <= params.max_tokens:
            pred = sorted(model.predict(tokens))
            pred = pred[-3:]
            _, senders = zip(*pred)
            if msg.sender in senders:
                yield {
                    'text': msg.text,
                    'sender': msg.sender,
                    'senders': senders,
                    'chat': [msg.to_dict() for msg in window(msgs, pos)]}


def quiz():
    questions = list(make_questions(get_messages()))
    with open('docs/quiz.json', 'w') as f:
        json.dump(questions[:2], f, ensure_ascii=False)


if __name__ == '__main__':
    quiz()
