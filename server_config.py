# this file is for running development server locally

import os
from route_config import *
from flask_socketio import SocketIO
app.debug = True
host = os.environ.get('IP', '0.0.0.0')
port = int(os.environ.get('PORT', 8080))
socketio = SocketIO(app)
ssl_context=('0.0.0.0.pem', '0.0.0.0-key.pem')


@socketio.on('message', namespace='/test')
def handle_message(data):
    print('received message: ' + data)


@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    print('Client connected')



# runs on https://0.0.0.0:5000
if __name__ == "__main__":
    socketio.run(app, host=host, port=5000, certfile='0.0.0.0.pem', keyfile='0.0.0.0-key.pem')
    # app.run(host=host, ssl_context=('0.0.0.0.pem', '0.0.0.0-key.pem'))
