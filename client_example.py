import numpy as np
from client import Client, JOINTS, DEFAULT_PORT
import sys

from test_bot import Robot

def run(cli):
    robot = Robot()
    
    while True:
        state=cli.get_state()
        
        j=robot.set_joints(state)

        cli.send_joints(j)


def main():
    name='Example Client'
    if len(sys.argv)>1:
        name=sys.argv[1]

    port=DEFAULT_PORT
    if len(sys.argv)>2:
        port=sys.argv[2]

    host='localhost'
    if len(sys.argv)>3:
        host=sys.argv[3]

    cli=Client(name, host, port)
    run(cli)


if __name__ == '__main__':
    '''
    python client_example.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'
    
    To run the one simulation on the server, run this in 3 separate command shells:
    > python client_example.py player_A
    > python client_example.py player_B
    > python server.py
    
    To run a second simulation, select a different PORT on the server:
    > python client_example.py player_A 9544
    > python client_example.py player_B 9544
    > python server.py -port 9544    
    '''

    main()
