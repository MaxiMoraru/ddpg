
import math


JOINT_FORCE=400
TABLE_HEIGHT=1.0
TABLE_THICKNESS=0.08
TABLE_LENGTH=2.4
TABLE_WIDTH=1.4
DT=1.0/50
JOINTS=11
DIST_THRESHOLD=3*TABLE_LENGTH
STATE_DIMENSION=37

def get_neutral_joint_position():
        jp=[0.0]*JOINTS
        jp[0]=-0.3
        jp[2]=math.pi
        a=math.pi/3.8
        jp[5]=a
        jp[7]=a
        jp[9]=math.pi/3.5 
        jp[10]=math.pi/2
        return jp

class AutoPlayerInterface:
    def __init__(self):
        jp=get_neutral_joint_position()
        self.stance=[
                self.make_stance(0.3, 0, 0.3, 0.3, -0.6-0.2),
                self.make_stance(0.3, 2.7, -1.3, -1.8, +0.4-0.1),
                self.make_stance(0.3, 0.4, -1, -1.3, +2)]
        self.stance_height=[0.17, 0.17, 0.6]
        self.stance_dy=[-0.32, +0.08, -0.35]
        self.chosen_stance=0
        self.freeze_stance=0.0

    def make_stance(self, a, j3, j5, j7, j9):
        jp=get_neutral_joint_position()
        jp[3] += a*j3
        jp[5] += a*j5
        jp[7] += a*j7
        jp[9] += a*j9 
        return jp

    def update(self, state):
        self.freeze_stance-=DT
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        if vy>0 or state[31]:
            jp=self.stance[self.chosen_stance].copy()
            jp[1]=bx
            return jp
        self.choose_stance(state)
        jp=self.stance[self.chosen_stance].copy()
        #dy=self.stance_dy[self.chosen_stance]
        self.choose_position(state, jp)
        return jp

    def choose_stance(self, state):
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        dist=math.hypot(px-bx, py-by, pz-bz)
        if not state[28] or dist<0.1 or self.freeze_stance>0.0:
            return
        g=9.81
        d=0.05
        minz=bz
        while by>-0.5 and bz>0.0:
            by+=vy*d
            bz+=vz*d
            vz-=g*d
            minz=min(minz,bz)
        curr=self.chosen_stance
        if minz>0.35:
            self.chosen_stance=2
        elif by>0 and by<0.30 and not state[30]:
            self.chosen_stance=1
        elif by<TABLE_LENGTH*0.5 and by>0.90 and not state[30]:
            self.chosen_stance=1
        elif by>0 and state[30]:
            self.chosen_stance=1
        elif by<-0.2:
            self.chosen_stance=0
        elif curr==2:
            self.chosen_stance=0
        if self.chosen_stance!=curr:
            self.freeze_stance=0.75

    def choose_position(self, state, jp):
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        dist=math.hypot(px-bx, py-by, pz-bz)
        vel=math.hypot(vx, vy, vz)
        if state[27] or vel<0.05:
            jp[1]=bx
            return
        extra_y=0.0
        if dist<vel*1.5*DT:
            extra_y=0.3
        d=0.05
        g=9.81
        while vz>0 or bz+d*vz>=pz:
            bx+=d*vx
            by+=d*vy
            bz+=d*vz
            vz-=d*g
        jp[1]=bx
        dy=py-state[0]
        jp[0]=by-dy+extra_y
        jp[10]-=(bx*0.3+vx*0.01)