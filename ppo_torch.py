import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        # log probabilities
        self.probs = []
        # values calculated by critic
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # get all batch chunks
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        # clear memory after episode
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    # alpha = learning rate
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        # defining actor network
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            # softmax takes care that we are dealing with probabilities. The sum must be 1.
            nn.Softmax(dim=-1)
        )
        # lr = learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # define forward action in the NN
    def forward(self, state):
        # get the distribution from the state based on the softmax ouput of the actor network
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    # alpha = learning rate
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        # defining actor network
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        # lr = learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    # define forward action in the NN
    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    # alpha = learning rate
    # gamma = discount factor
    # N = horizon = number of steps before we perform an update
    # gae_lambda = lambda value for the Generalized Advantage Estimation (GAE) .
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... load models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    # Implementations of equations 11 and 12 of https://arxiv.org/pdf/1707.06347.pdf
    def learn(self):
        for _ in range(self.n_epochs):
            # arr = arrays
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # for each timestep t
            for t in range(len(reward_arr) - 1):
                # discount = gae_lambda * gamma
                discount = 1
                # a_t = advantage at time t
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    # this part takes care of the multiplication power of the term (gamma*gae_lambda)
                    discount *= self.gamma * self.gae_lambda
                    advantage[t] = a_t
                advantage = T.tensor(advantage).to(self.actor.device)
                for batch in batches:
                    # this is the denominator of the probability ratio r_t(theta) from the paper.
                    states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                    old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                    actions = T.tensor(action_arr[batch]).to(self.actor.device)
                    # this is the numerator of the probability ration r_t(theta) from the paper.
                    dist = self.actor(states)
                    critic_value = self.critic(states)
                    critic_value = T.squeeze(critic_value)

                    new_probs = dist.log_prob(actions)
                    # Returns a new tensor with the exponential of the elements of the input tensor
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    # this is equation 6 from the paper
                    weighted_probs = advantage[batch] * prob_ratio

                    weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * \
                                             advantage[batch]
                    # this is equation 7 from the paper
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns - critic_value) ** 2
                    # L_t ^(VF) in equation 9
                    critic_loss = critic_loss.mean()

                    # c_1 = 0.5 in equation 9
                    total_loss = actor_loss + 0.5 * critic_loss
                    # Sets the gradients of all optimized torch.Tensor to zero.
                    # In PyTorch, for every mini-batch during the training phase, we typically want to explicitly set
                    # the gradients to zero before starting to do backpropragation
                    # (i.e., updating the Weights and biases) because PyTorch accumulates the
                    # gradients on subsequent backward passes. This accumulating behaviour is convenient while
                    # training RNNs or when we want to compute the gradient of the
                    # loss summed over multiple mini-batches.
                    # So, the default action has been set to accumulate (i.e. sum)
                    # the gradients on every loss.backward() call.
                    # Because of this, when you start your training loop, ideally you should zero out the gradients
                    # so that you do the parameter update correctly. Otherwise, the gradient would be a
                    # combination of the old gradient, which you have already used to update your model parameters,
                    # and the newly-computed gradient. It would therefore point in some other direction than the
                    # intended direction towards the minimum (or maximum, in case of maximization objectives).
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    total_loss.backward()
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()
        self.memory.clear_memory()
