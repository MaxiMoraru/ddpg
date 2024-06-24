import numpy as np
import math
from ikpy.chain import Chain
import transforms3d
TABLE_HEIGHT=1.0
TABLE_THICKNESS=0.08
TABLE_LENGTH=2.4
TABLE_WIDTH=1.4
G = 9.81
D = 0.05
# table center pos = [x = 0, y=0, z =TABLE_HEIGHT]
# table perimeters = 
#  1---|---2            
#  |       |        
# -|-[0,0]-|-      
#  |       |        
#  4---|---3       

# [(z)]      = up
# /\
# ||
# (y)        = forward
# (x)==>     = right



# 1 = [-TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT]
# 2 = [TABLE_LENGTH/2, -TABLE_WIDTH/2, TABLE_HEIGHT]
# 3 = [TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT]
# 4 = [-TABLE_LENGTH/2, TABLE_WIDTH/2, TABLE_HEIGHT]


"""
            State list
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


"""


# Joint limits
joint_limits = {
    0: (-0.3, 0.3),                     # Forward-Backward Slider
    1: (-0.8, 0.8),                     # Left-Right Slider
    2: (-np.inf, np.inf),         # Rotation around Z (limited to +/- 360 degrees)
    3: (-math.pi/2, math.pi/2),         # Pitch of the first arm link
    4: (-np.inf, np.inf),         # Roll of the first arm link
    5: (-3*math.pi/4, 3*math.pi/4),     # Pitch of the second arm link
    6: (-np.inf, np.inf),         # Roll of the second arm link
    7: (-3*math.pi/4, 3*math.pi/4),     # Pitch of the third arm link
    8: (-np.inf, np.inf),         # Roll of the third arm link
    9: (-3*math.pi/4, 3*math.pi/4),     # Pitch of the paddle
    10: (-np.inf, np.inf)         # Roll of the paddle 
}
class Robot:

    def __init__(self):
        self.robot = Chain.from_urdf_file("test_bot.urdf")
        self.joints = np.zeros((9,))
        self.joint_limits = joint_limits

    def set_joints(self,state):
        joints_to_send = j=np.zeros((11,))
        # if game is waiting send 0 to all joints
        if state[26] == 1:
            return joints_to_send
        
        # if game is waiting for opponent service send 0 to all joints
        if state[27] == 1:
            return joints_to_send
        
        # if you hit the ball and the opponent has not touched the ball yet
        # move the base to the center of the table
        if state[30] == 1 and state[33] == 0:
            current_base_pos = state[0:2]
            target_base_pos = [current_base_pos[0], 0.0]
            # if the base is not in the center of the table
            if current_base_pos[0] != target_base_pos[0] and current_base_pos[1] != target_base_pos[1]:
                if abs(current_base_pos[1]) > 0.3:
                    joints_to_send[1] = 0.3
                else:
                    joints_to_send[1] = -current_base_pos[1]
                joints_to_send[0] = -0.01
        
        #if the opponent hit the ball and the ball hasnt touched your side of the table
        # move the base to ward the ball x position
        if state[31] == 0 and state[32] == 1:
            current_base_pos = state[0:2]
            ball_velocity_xy = state[20:22]
            ball_normal_xy = ball_velocity_xy / np.linalg.norm(ball_velocity_xy)
            joints_to_send[0] = ball_normal_xy[0] * 0.01
            joints_to_send[1] = ball_normal_xy[1] * 0.01

        #if the ball touched your side of the table and you have not touched the ball yet
        # get the prediction of the ball hit point and the paddle normal
        # move the paddle to the hit point

        if state[30] == 1 and state[31] == 0:
            predict_point, paddle_normal = calculate_prediction_and_paddle_normal(state)
            paddle_pos = state[11:14]
            paddle_normal = state[14:17]

            joints = state[2:11]

            # use the inverse kinematics
            # Create target frame for IK
            target_position = predict_point
            # Convert paddle normal to rotation matrix (this step is crucial!)
            target_rotation = transforms3d.quaternions.mat2quat(paddle_normal.reshape(3, 3))
            target_frame = np.eye(4)  # 4x4 identity matrix
            target_frame[:3, :3] = transforms3d.quaternions.quat2mat(target_rotation)  
            target_frame[:3, 3] = target_position

            # Solve for joint angles using IK
            joint_angles = self.robot.inverse_kinematics(target_frame)

            # Apply joint limits (optional, but recommended)
            for i, angle in enumerate(joint_angles):
                if angle < self.joint_limits[i][0] or angle > self.joint_limits[i][1]:
                    # Handle joint limit violation (e.g., clamp to limit, raise error, etc.)
                    print(f"Warning: Joint {i} limit violated ({angle:.2f} outside {self.joint_limits[i]}).")

            # Assign the solved joint angles to the joints_to_send array
            joints_to_send[2:11] = joint_angles  # Exclude base joints (0 and 1)

        return joints_to_send




def calculate_prediction_and_paddle_normal(state):
    """
    Calculates the predicted ball hit position and the desired paddle normal for a successful shot.

    Args:
        state: A list containing the current state of the game.

    Returns:
        predict_point: A numpy array representing the (x, y, z) coordinates of the predicted hit point.
        paddle_normal: A numpy array representing the (x, y, z) normal vector for the paddle.
    """

    # Extract relevant state values
    ball_x, ball_y, ball_z = state[17], state[18], state[19]
    ball_v_x, ball_v_y, ball_v_z = state[20] * D, state[21] * D, state[22] * D
    paddle_x, paddle_y = state[11], state[12]  # Paddle z not used in this calculation

    # Calculate distances (ignoring z since paddle is assumed to adjust height perfectly)
    d_By_Py = abs(ball_y - paddle_y)
    d_Bx_Px = abs(ball_x - paddle_x)

    # Time to reach the paddle's y-position
    t = d_By_Py / (2 * abs(ball_v_y))

    # Predicted hit point coordinates
    predict_x = ball_x + ball_v_x * t
    predict_y = ball_y - d_By_Py / 2  # Midpoint between ball and paddle in y
    predict_z = ball_z + ball_v_z * t - 0.5 * G * t**2

    predict_point = np.array([predict_x, predict_y, predict_z])

    # Vectors for current and desired ball directions
    ball_current_vector = np.array([ball_v_x, ball_v_y])
    ball_desired_vector = np.array([TABLE_LENGTH / 4 - predict_x, -predict_y])

    # Calculate the bisector (average direction for reflection) and normalize to get the paddle normal
    bisector_vector = ball_current_vector + ball_desired_vector
    paddle_normal_xy = bisector_vector / np.linalg.norm(bisector_vector)

    # Calculate the z component of the paddle normal
    distance_ball_paddle_xy = np.linalg.norm(predict_point[0:2] - np.array([paddle_x, paddle_y]))
    paddle_normal_z = (G * distance_ball_paddle_xy) / (2 * ball_v_x**2)

    paddle_normal = np.array([paddle_normal_xy[0], paddle_normal_xy[1], paddle_normal_z])

    return predict_point, paddle_normal