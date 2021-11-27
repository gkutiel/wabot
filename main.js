addEventListener('DOMContentLoaded', async () => {
    qid = 236

    const msg = await (await fetch(`msgs/${qid}.json`)).json()
    document.getElementById('q').innerHTML = msg.text
    for (let i = 1; i <= 3; i++) {
        const label = document.getElementById(`l${i}`)
        const sender = msg.senders[i - 1]
        label.innerHTML = sender
        label.setAttribute('correct', 'no')
        if (msg.sender == sender) {
            label.setAttribute('correct', 'yes')
        }
    }

    const answers = document.querySelectorAll('input')

    function checkAnswers() {
        answers.forEach(answer => {
            answer.disabled = true
        })
        document.getElementById('chat').style.opacity = 1
    }

    answers.forEach(answer => {
        answer.addEventListener('change', checkAnswers)
    })

    const msgs = await (await fetch(`chats/${qid}.json`)).json()

    const chat = document.getElementById('chat')

    for (let m of msgs) {
        const sender = document.createElement('div')
        sender.className = 'sender'
        sender.innerHTML = m.sender

        const text = document.createElement('div')
        text.innerHTML = m.text

        const time = document.createElement('time')
        time.innerHTML = m.time

        const msg = document.createElement('div')
        msg.className = 'msg'

        msg.appendChild(time)
        msg.appendChild(sender)
        msg.appendChild(text)

        chat.appendChild(msg)
    }
})

