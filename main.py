"""
Based on the Chat application from menard-codes: https://github.com/menard-codes/FlaskChatApp
"""

from flask import Flask, request, render_template, redirect, url_for, session
from flask_socketio import SocketIO, join_room, leave_room, send

from llm import create_llm, model_interfaces
from utils import generate_room_code

import click


app = Flask(__name__)
app.config['SECRET_KEY'] = 'SDKFJSDFOWEIOF'
socketio = SocketIO(app)


rooms = {}
loaded_llms = {}


def get_settings_from_page():
    res = {
        'llm_name': request.form.get("llm_name"),
        "sys_prompt": request.form.get("sys_prompt"),
    }

    try:
        res["max_new_tokens"] = int(request.form.get("max_new_tokens"))
    except TypeError:  # If the max_new_tokens is not an integer, leave it blank
        pass

    return res


@app.route('/', methods=["GET", "POST"])
def home():
    session.clear()

    if request.method == "POST":

        room_id = generate_room_code(6, list(rooms.keys()))
        new_room = {
            'messages': [],
            'settings': get_settings_from_page()
        }
        rooms[room_id] = new_room

        session['room'] = room_id

        return redirect(url_for(f'room', room_id=room_id))
    else:
        return render_template(
            'home.html', llm_models=model_interfaces.keys(), rooms=rooms, loaded_llms=loaded_llms
        )


@app.route('/room/<room_id>', methods=["GET", "POST"])
def room(room_id):
    session["room"] = room_id

    if room_id is None or room_id not in rooms:
        return redirect(url_for('home'))

    current_room = rooms[room_id]

    if request.method == "POST":
        current_room["settings"] = get_settings_from_page()

    llm_name = current_room["settings"]["llm_name"]
    if llm_name not in loaded_llms:
        loaded_llms[llm_name] = create_llm(llm_name)

    return render_template(
        'room.html',
        llm_models=model_interfaces.keys(),
        rooms=rooms,
        current_room=current_room,
        loaded_llms=loaded_llms
    )


@socketio.on('connect')
def handle_connect():
    room_id = session.get('room')

    if room_id is None:
        return
    if room_id not in rooms:
        leave_room(room_id)

    join_room(room_id)


def add_message_to_room(room_id, room_messages, text, sender):
    message = {
        "sender": sender,
        "message": text
    }
    send(message, to=room_id)
    room_messages.append(message)


@socketio.on('message')
def handle_message(payload):
    room_id = session.get('room')

    if room_id not in rooms:
        return
    current_room = rooms[room_id]
    room_messages = current_room["messages"]
    llm_name = current_room["settings"]["llm_name"]

    add_message_to_room(room_id, room_messages, payload["message"], "user")

    llm = loaded_llms.get(llm_name)
    if llm is None:
        add_message_to_room(room_id, room_messages, "ERROR: The requested LLM has not been loaded", "error")

    res = llm.get_answer_to(
        room_messages,  # Last message has already been added above
        current_room["settings"]
    )

    add_message_to_room(room_id, room_messages, res, llm_name)


@socketio.on('disconnect')
def handle_disconnect():
    room_id = session.get("room")
    leave_room(room_id)


@click.command()
@click.option('--host', default="127.0.0.1")
@click.option('--port', default=5000)
@click.option('--debug/--no-debug', default=True)
def main(host, port, debug):
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()
