import json
import random

from datetime import timedelta
from tqdm import tqdm
from typing import List
from wabot.Params import Params
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.parser import Msg, get_messages


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


def quiz(folder, model_ckpt, chat_txt, num_questions=10):
    questions = list(make_questions(get_messages()))
    random.shuffle(questions)
    questions = questions[:num_questions]
    with open(f'docs/{folder}/index.html', 'w') as f:
        f.write(r'''
<!DOCTYPE html>
<html>
<style>
    * {
        box-sizing: border-box;
    }

    html,
    body {
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 0;
        border: 1px solid transparent;
    }

    body {
        background-color: whitesmoke;
    }

    main {
        overflow: hidden;
        padding: 2em;
        width: 30em;
        margin: 10em auto;
        background-color: white;
        box-shadow: 0 0 0.3em #aaa;
    }

    #q {
        font-weight: bold;
    }

    .answers {
        line-height: 1.5em;
    }

    .answer>label {
        line-height: 1.5em;
        cursor: pointer;
    }

    .answer:hover {
        background-color: #eee;
    }
2
    input[type=radio] {
        cursor: pointer;
    }

    input[type=radio]:disabled {
        display: none;
    }

    input[type=radio]:disabled+label {
        cursor: default;
    }

    :disabled+label[correct='no']::before {
        content: '❌';
        margin-left: .5em;
    }

    :disabled+label[correct='yes']::before {
        content: '✔️';
        margin-left: .5em;
    }

    #chat {
        max-height: 80em;
        margin: 2em 0;
        overflow: auto;
        font-size: .8em;
        opacity: 0;
        /* transition: opacity 3s; */
    }

    .msg {
        margin: .5em 0;
        padding: .3em .5em;
        background-color: #f0fff3;
        border: 1px solid #eee;
        border-radius: .5em;
    }

    time {
        color: #999;
        font-size: .8em;
        float: left;
    }

    .sender {
        font-weight: bold;
    }

    #score {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
    }
</style>

<head>
    <meta charset='utf-8'>
    <meta http-equiv='X-UA-Compatible' content='IE=edge'>

    <title>מי אמר למי?</title>

    <meta property="og:title" content="מי אמר למי?">
    <meta property="og:site_name" content="מי אמר למי?">
    <meta property="og:url" content="">
    <meta property="og:description" content="מי אמר למי?.">
    <meta property="og:type" content="website">
    <meta property="og:image" content="https://gkutiel.github.io/wabot/favicon.png">

    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <link rel="icon" type="image/png" href="https://gkutiel.github.io/wabot/favicon.png" />


</head>

<body dir='rtl'>
    <main>
        <p>מי אמר/ה?</p>
        <p id='q'></p>

        <div id='answers'>
        </div>

        <div id="chat"></div>

        <div class="next">
            <button id='next'>שאלה הבאה</button>
        </div>
        <div id="score"></div>
    </main>
</body>

<script>
    function $(id) {
        return document.getElementById(id);
    }

    function e(html) {
        const div = document.createElement('div')
        div.innerHTML = html
        return div.firstElementChild
    }

    class Question {
        constructor(text) {
            this.e = $('q');
        }
        set text(text) {
            this.e.innerText = text;
        }
    }

    class Answer {
        static id = 0;
        constructor(cb) {
            const id = Answer.id++;
            this.good = false
            this.e = e(`
                <div class="answer">
                    <input type='radio' id='${id}' name='answer' />
                    <label for='${id}' correct='no'></label>
                </div>
            `)
            $('answers').appendChild(this.e)

            this.e.addEventListener('change', () => {
                $('chat').style.opacity = 1;
                cb(this.correct);
            })

            this.label = this.e.querySelector('label')
            this.input = this.e.querySelector('input')
        }

        set text(text) {
            this.label.innerText = text;
            this.input.disabled = false;
            this.input.checked = false;
        }

        set correct(correct) {
            this.good = correct;
            this.label.setAttribute('correct', correct ? 'yes' : 'no')
        }

        get correct() {
            return this.good;
        }

        disable() {
            this.input.disabled = true;
        }
    }

    class Chat {
        constructor() {
            this.e = $('chat');
        }
        set msgs(msgs) {
            this.e.style.opacity = 0;
            this.e.innerHTML = '';
            for (const msg of msgs) {
                this.e.appendChild(e(`
                    <div class='msg'>
                        <time>${msg.time}</time>
                        <span class='sender'>${msg.sender}</span>
                        <div>${msg.text}</div>
                    </div>
                `))
            }
        }
    }


    class Quiz {
        constructor(quiz) {
            this.score = 0;
            this.qid = 0;
            this.quiz = quiz;

            this.scoreDiv = $('score');
            this.question = new Question();

            this.answers = [
                new Answer(this.on_answer),
                new Answer(this.on_answer),
                new Answer(this.on_answer)];

            this.chat = new Chat();
            this.nextButton = $('next');
            this.nextButton.addEventListener('click', this.next);
        }

        nextQuestion() {
            return this.quiz[this.qid++];
        }

        on_answer = (correct) => {
            if (correct) {
                this.score++;
            }

            for (let answer of this.answers) {
                answer.disable();
            }

            this.nextButton.disabled = false;
            this.scoreDiv.innerText = `${this.qid}/${this.quiz.length}`;
        }

        next = () => {
            try {
                const q = this.nextQuestion();
                this.nextButton.disabled = true;
                this.question.text = q.text;
                this.chat.msgs = q.chat;

                for (let i = 0; i < this.answers.length; i++) {
                    this.answers[i].text = q.senders[i];
                    this.answers[i].correct = q.senders[i] === q.sender;
                }
            } catch (e) {
                const score = (this.score / this.qid * 100).toFixed(0)
                this.scoreDiv.innerText = `תוצאה סופית: ${score}%`;
            }
        }
    }
    ''')

        f.write(f'const quiz = new Quiz({json.dumps(questions, ensure_ascii=False)})')

        f.write(r'''

    quiz.next()
</script>

</html>
        ''')


if __name__ == '__main__':
    quiz()
