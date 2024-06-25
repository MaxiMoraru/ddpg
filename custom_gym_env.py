import math
import numpy as np
from gymnasium.spaces import Box
# table dimensions
TABLE_HEIGHT=1.0
TABLE_THICKNESS=0.08
TABLE_LENGTH=2.4
TABLE_WIDTH=1.4
TABLE_MIDDLE = 1.0

G = 9.81  # Gravity constant
D = 0.05  # Time step

# table center pos = [x = 0, y=1, z = 0]
# table perimeters = 
#  1---|---2
#  |       |
# -|-[0,1]-|-
#  |       |
#  4---|---3
TABLE_MIN_X = -TABLE_WIDTH/2
TABLE_MAX_X = TABLE_WIDTH/2
TABLE_MIN_Y = TABLE_MIDDLE -TABLE_LENGTH/2
TABLE_MAX_Y = TABLE_MIDDLE + TABLE_LENGTH/2


        

    


class CustomEnv():
    def __init__(self, client):


         # Define your observation space
        self.observation_space = Box(
            low=np.array(
                [-np.inf] * 11 +  # 0-10 Joint positions  
                [-np.inf] * 3 +   # 11-13 Paddle center position 
                [-1.0] * 3 +      # 14-16 Paddle normal versor
                [-np.inf] * 3 +   # 17-19 Ball position
                [-np.inf] * 3 +   # 20-22 Ball velocity
                [-np.inf] * 3 #+   # 23-25 Predicted hit point
                #[-1.0] * 3        # 26-28 Desired paddle normal
            ),
            high=np.array(
                [np.inf] * 11 +  # 0-10 Joint positions
                [np.inf] * 3 +   # 11-13 Paddle center position
                [1.0] * 3 +      # 14-16 Paddle normal versor
                [np.inf] * 3 +   # 17-19 Ball position
                [np.inf] * 3 +   # 20-22 Ball velocity
                [np.inf] * 3 #+   # 23-25 Predicted hit point
                #[1.0] * 3        # 26-28 Desired paddle normal
            ),
            dtype=float
        )
            

        
            
        
        # Define your action space (assuming it's already defined similarly)
        self.action_space = Box(
            low=np.array([-0.3, -0.8, -np.inf, -math.pi/2, -np.inf, 
                        -3*math.pi/4, -np.inf, -3*math.pi/4, -np.inf,
                        -3*math.pi/4, -np.inf]),  # Low bounds as np.ndarray
            high=np.array([0.3, 0.8, np.inf, math.pi/2, np.inf, 
                        3*math.pi/4, np.inf, 3*math.pi/4, np.inf,
                        3*math.pi/4, np.inf]),    # High bounds as np.ndarray
            dtype=float
        )


        # Initialize your client for communication with the server
        self.client = client
        
        



    def get_state(self):
        state = self.client.get_state()

        # remove the last value of the state (dont need it)
        state = state[:-1]



        # calculate the predicted hit point and the desired paddle normal
        predict_point, paddle_normal = calculate_predict_point(state)
        # add the predicted hit point and the desired paddle normal to the state
        # if the predicted point is [0, 0, 0] then use the current state as the predicted point
        if predict_point == [0, 0, 0]:
            predict_point = state[11:14]
        state = np.append(state, predict_point)
        #state = np.append(state, paddle_normal)

        #print("----------------------------")
            
        return state

    def extract_state(self,state):
        # Extract the used values of the state (from 0 to 22 and 37 to 39)
        state = np.append(state[0:23], state[37:40])
        return state

    def reset(self):
        # Reset the environment, return initial state
        # Get the state from the client
        state = self.get_state()
        # if the game is waiting, the robot cannot move
        while state[26] == 1:
            # Send a command to stop the robot
            self.client.send_joints([0,0,0,0,0,0,0,0,0,0,0])
            # Get the state from the client
            state = self.get_state()
        # extract the used values of the state (from 0 to 22)
        state = self.extract_state(state)
        
        return state
        

    def step(self, action):
        # Take a step in the environment with the given action
        self.client.send_joints(action)
        # Get the next state, reward, done, and info
        next_state = self.get_state()
        reward = self.calculate_reward(next_state, action)  # Implement your reward function
        done = self.is_done(next_state)  # Implement your termination condition

        info = {}  # Additional information, if needed

        # extract the used values of the state (from 0 to 22)
        next_state = self.extract_state(next_state)

        return next_state, reward, done ,info

    def render(self, mode='human'):
        # Optional: Render the environment for visualization
        pass

    def close(self):
        # Clean up resources
        self.client.close()

    def seed(self, seed=None):
        # Optional: Set random seed for reproducibility
        pass




    def calculate_reward(self, state, action):

        """
            OLD State list
            Index       Description
            0-10        Current joint positions
            11-13       Paddle center position (x,y,z)
            14-16       Paddle normal versor (x,y,z)
            17-19       Current ball position (x,y,z)
            20-22       Current ball velocity (x,y,z)
            23-25       Opponent paddle center position (x,y,z)
            26          Game waiting, cannot move (0=no, 1=yes)
            27          Game waiting for opponent service (0=no, 1=yes)
            28          Game playing (i.e., not waiting) (0=no, 1=yes)
            29          Ball in your half-field (0=no, 1=yes)
            30          Ball already touched your side of table (0=no, 1=yes)
            31          Ball already touched your robot (0=no, 1=yes)
            32          Ball in opponent half-field (0=no, 1=yes)
            33          Ball already touched opponent side of table (0=no, 1=yes)
            34          Your score
            35          Opponent score
            36          Simulation time
            37-39       Predicted hit point (x,y,z)
            40-42       Desired paddle normal (x,y,z)

            NEW State list (used to send to the AI model)
            Index       Description
            0-10        Current joint positions
            11-13       Paddle center position (x,y,z)
            14-16       Paddle normal versor (x,y,z)
            17-19       Current ball position (x,y,z)
            20-22       Current ball velocity (x,y,z)
            23-25       Predicted hit point (x,y,z)
            26-28       Desired paddle normal (x,y,z)

        """
        base_position = state[0:2]
        paddle_position = state[11:14]
        paddle_normal = state[14:17]
        ball_position = state[17:20]
        ball_velocity = state[20:23]
        opponent_paddle_position = state[23:26]
        game_waiting = state[26]
        game_waiting_opponent = state[27]
        game_playing = state[28]
        ball_in_your_half = state[29]
        ball_touched_your_side = state[30]
        ball_touched_robot = state[31]
        ball_in_opponent_half = state[32]
        ball_touched_opponent_side = state[33]
        your_score = state[34]
        opponent_score = state[35]
        simulation_time = state[36]
        predicted_hit_point = state[37:40]
        #desired_paddle_normal = state[40:43]

        reward = 0


        # penalize the robot for moving
        reward -= np.linalg.norm(action)* 0.1

        # reward for ball touching the opponent side of the table
        if ball_touched_opponent_side == 1:
            reward += 10
            
        # reward for hitting the ball towards the opponent
        if ball_velocity[1] > 0:
            reward += 1
        # prenalize for having the paddle far from the predicted hit point
        reward -= abs(predicted_hit_point[0] - paddle_position[0])*0.1
        reward -= abs(predicted_hit_point[1] - paddle_position[1])*0.1
        reward -= abs(predicted_hit_point[2] - paddle_position[2])*0.1



        return reward 
    
    




    def is_done(self,state):
        done = False
        # if either player scored the episode is done
        if state[26] == 1:
            done = True
        return done

"""
# Example usage:
if __name__ == '__main__':
    env = CustomEnv()
    observation = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Replace with your agent's action
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
"""

def calculate_predict_point(state):
    # check for ball speed so its not zero
    if np.linalg.norm(state[20:23])>0.01: # ball speed is not zero
        #print("Ball speed is not zero")
        # check if the ball is in the air
        if state[19] > 0: # ball is in the air
           #print("Ball is in the air")
            # check the direction of the ball
            if state[21] > 0: # ball is moving towards the opponent
                #print("Ball is moving towards the opponent")
                return [0, 0, 0], [0, 0, 0] 
            else: # ball is moving towards you
                #print("Ball is moving towards you")
                # check if the ball aready touched your side of the table
                if state[30] == 0: # ball hasnt touched your side of the table
                    #print("Ball hasnt touched your side of the table")
                    x,y,z = find_impact_point(state[17:20], state[20:23])
                    # check if the impact point is within your side of the table
                    if x >= TABLE_MIN_X and x <= TABLE_MAX_X and y >= TABLE_MIN_Y and y <= TABLE_MIDDLE: # ball is going to hit your side of the table
                        #print("Ball is going to hit your side of the table")
                        # predict the new impact point
                        ball_velocity = state[20:23]
                        ball_velocity_after_impact = [ball_velocity[0], ball_velocity[1], -ball_velocity[2]]
                        # find the point where the ball is the highest
                        x_highest, y_highest, z_highest = find_highest_point([x,y,z], ball_velocity_after_impact)                     
                        return [x_highest, y_highest, z_highest], [0, 0, 0]
                    else: # ball wont hit your side of the table
                        #print("Ball wont hit your side of the table")
                        return [0, 0, 0], [0, 0, 0]  
                else: # ball has already touched your side of the table
                    #print("Ball has already touched your side of the table")
                    x_highest, y_highest, z_highest = find_highest_point(state[17:20], state[20:23])
                    # check if the ball went beyond the predicted hit point
                    if y_highest > state[18]: # ball went beyond the predicted hit point
                        # return the current ball position
                        return [state[17],state[18],state[19]], [0, 0, 0]
                    # else return the predicted hit point
                    return [x_highest, y_highest, z_highest], [0, 0, 0]
        else: # ball is below the table
            #print("Ball is below the table")
            return [0, 0, 0], [0, 0, 0]
    else: # ball speed is zero
        #print("Ball speed is zero")
        return [0, 0, 0], [0, 0, 0]

        

def find_impact_point(p0, v0, g = 9.81):
    x0, y0, z0 = p0
    vx0, vy0, vz0 = v0

    # Gravity is negative, indicating it is acting downward
    g_z = -g

    # Coefficients for the quadratic equation: 0.5 * g_z * t^2 + vz0 * t + (z0 - z_table) = 0
    a = 0.5 * g_z
    b = vz0
    c = z0 

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("No real solution for the time of impact. Check your initial conditions and parameters.")

    # Calculate the two possible solutions for t
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Choose the positive time
    t_impact = max(t1, t2)

    # Calculate the x and y coordinates at time of impact
    x_hit = x0 + vx0 * t_impact
    y_hit = y0 + vy0 * t_impact

    return x_hit, y_hit, 0

def find_highest_point(p0, v0, g = 9.81):
    x0, y0, z0 = p0
    vx0, vy0, vz0 = v0

    # Gravity is negative, indicating it is acting downward
    g_z = -g

    # Time at which the ball reaches the highest point (when vz = 0)
    t_highest = -vz0 / g_z

    # Calculate the x, y, and z coordinates at time t_highest
    x_highest = x0 + vx0 * t_highest
    y_highest = y0 + vy0 * t_highest
    z_highest = z0 + vz0 * t_highest + 0.5 * g_z * t_highest**2

    return x_highest, y_highest, z_highest
