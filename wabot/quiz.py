from tqdm import tqdm
from typing import List, cast
from torch import Tensor
from dataclasses import dataclass
from wabot.Params import Params
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.parser import Msg, get_messages, get_sessions


def quiz():
    params = Params()
    model = SimpleSenderPredictor.load_from_checkpoint('model.ckpt')
    msgs = get_messages()

    with open('quiz.txt', 'w', encoding='utf-8') as f:
        for msg in tqdm(msgs):
            tokens = model.text_encoder.tokenize(msg.text)
            if params.min_tokens <= len(tokens) <= params.max_tokens:
                pred = sorted(model.predict(tokens))
                pred = pred[-3:]
                _, senders = zip(*pred)
                if msg.sender in senders:
                    print('=' * 80, file=f)
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
