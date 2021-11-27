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
        msg.dict()
        for msg
        in msgs[pos-max_size:pos+max_size]
        if max(msg.time, time) - min(msg.time, time) < delta
        and not '<Media omitted>' in msg.text]


def quiz():
    params = Params()
    model = SimpleSenderPredictor.load_from_checkpoint('model.ckpt')
    msgs = get_messages()

    with open('chat.json', 'w') as f:
        json.dump([msg.dict() for msg in msgs], f)

    with open('quiz.txt', 'w', encoding='utf-8') as f:
        for mid, msg in tqdm(enumerate(msgs)):
            tokens = model.text_encoder.tokenize(msg.text)
            if params.min_tokens <= len(tokens) <= params.max_tokens:
                pred = sorted(model.predict(tokens))
                pred = pred[-3:]
                _, senders = zip(*pred)

                if msg.sender in senders:
                    with open(f'msgs/{mid}.json', 'w') as j:
                        d = msg.dict()
                        d['senders'] = senders
                        json.dump(d, j)

                    with open(f'chats/{mid}.json', 'w') as j:
                        json.dump(window(msgs, mid), j)

                    print('=' * 80, file=f)
                    print(mid, file=f)
                    print('מי אמר/ה?', file=f)
                    print(file=f)
                    print(f'*{msg.text}*', file=f)
                    print(file=f)

                    for i, sender in enumerate(senders):
                        print(f'{i+1}. {sender}', file=f)

                    print(file=f)
                    print(msg.sender, file=f)


if __name__ == '__main__':
    quiz()
