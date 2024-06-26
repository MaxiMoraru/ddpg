from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import core as core
from client import Client

import custom_gym_env as cge
from bot import AutoPlayerInterface

MODEL_PATH = "Models/model.pt"
CHECKPOINT_PATH = "Models/checkpoints/"

#Hyperparameters
ACTOR_CRITIC = core.MLPActorCritic
SEED = 0
STEPS_PER_EPOCH = 10000
EPOCHS = 1000
REPLAY_SIZE = int(1e6)
GAMMA = 0.99
POLYAK = 0.995
PI_LR = 1e-4
Q_LR = 1e-3
BATCH_SIZE = 128
START_STEPS = 10000
UPDATE_AFTER = 400
UPDATE_EVERY = 50
ACT_NOISE = 0.1
NUM_TEST_EPISODES = 10
MAX_EP_LEN = 5000
CHECKPOINT = False

#Actor Critic
HID_SIZE = 350
LAYERS = 2



class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def save_model(ac, pi_optimizer, q_optimizer, path):
    torch.save({
        'ac_state_dict': ac.state_dict(),
        'pi_optimizer_state_dict': pi_optimizer.state_dict(),
        'q_optimizer_state_dict': q_optimizer.state_dict(),
    }, path)

def load_model(ac, pi_optimizer, q_optimizer, path):
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['ac_state_dict'])
    pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
    q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])



def ddpg(env_fn, actor_critic=ACTOR_CRITIC, ac_kwargs=dict(), seed=SEED, 
         steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, replay_size=REPLAY_SIZE, gamma=GAMMA, 
         polyak=POLYAK, pi_lr=PI_LR, q_lr=Q_LR, batch_size=BATCH_SIZE, start_steps=START_STEPS, 
         update_after=UPDATE_AFTER, update_every=UPDATE_EVERY, act_noise=ACT_NOISE, num_test_episodes=NUM_TEST_EPISODES, 
         max_ep_len=MAX_EP_LEN, logger_kwargs=dict(), save_freq=1, checkpoint_continue=CHECKPOINT):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)


    if checkpoint_continue:
        load_model(ac, pi_optimizer, q_optimizer, MODEL_PATH)
        ac.train()

    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    print('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)  
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        #print('LossQ= ', loss_q.item(), 'LossPi= ', loss_pi.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        # clip the action to the action space for each action with its own limit
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            print('Test Episode: ', j)
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            print('TestEpRet (reward) = ', ep_ret, 'TestEpLen= ', ep_len)


    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()
            #a = bot.update(o)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            print('EpRet (reward) = ', ep_ret, 'EpLen= ', ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            print('Epoch: ', epoch)

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                #logger.save_state({'env': env}, None)
                # save the model
                print('Saving model...')
                save_model(ac, pi_optimizer, q_optimizer, MODEL_PATH)
                

            print('Testing agent...')
            # Test the performance of the deterministic version of the agent.
            test_agent()


            print('----------------------------------')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load' , type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9543)
    parser.add_argument('--name', type=str, default='PingPong AI')
    args = parser.parse_args()

    #from spinup.utils.run_utils import setup_logger_kwargs
    #logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    cli=Client(args.name, args.host, args.port)
    env = cge.CustomEnv(cli)
    if args.load:
        ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=[HID_SIZE]*LAYERS)
        ac.load_state_dict(torch.load(MODEL_PATH))
        ac.eval()
        while True:
            o = env.reset()
            while True:
                a = ac.act(torch.as_tensor(o, dtype=torch.float32))
                o, r, d, _ = env.step(a)
                if d:
                    break
    else:
        ddpg(lambda : env, ac_kwargs=dict(hidden_sizes=[HID_SIZE]*LAYERS))

