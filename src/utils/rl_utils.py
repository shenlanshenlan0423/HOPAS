# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：utils.py
@IDE ：PyCharm
"""
from definitions import *
from src.eval.agent_eval import run_evaluate
from src.utils.utils import DatasetReconstruction


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class VAnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(self.fc1(x))
        V = self.fc_V(self.fc1(x))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = []
        for _ in range(batch_size):
            idx = random.choice(range(len(self.buffer)))
            transitions.append(self.buffer[idx])
            self.buffer.pop(idx)  # 不放回采样
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

    def get_all_samples(self):
        all_transitions = self.buffer
        state, action, reward, next_state, done = zip(*all_transitions)
        return state, action, reward, next_state, done


def SIT_heparin_response(value, target_interval):
    """
    For monitering the Platelet and aPTT to check the effectiveness of heparin
    """
    denominator = (target_interval[1] - target_interval[0]) * 0.1
    f = 2 / (1 + torch.exp(-(value - target_interval[0]) / denominator)) - 2 / (1 + torch.exp(-(value - target_interval[1]) / denominator)) - 1
    return f


def prepare_replay_buffer(tensor_tuple, args):
    """
    Note: For agents
    Platelets: [50, 150] -> [-0.86322, -0.55927]
    aPTT: [60, 100] -> [0.13923, 0.60281]
    """
    replay_buffer = ReplayBuffer()
    Sepsis_data = DatasetReconstruction(tensor_tuple)
    for patient_idx in tqdm(range(len(Sepsis_data))):
        states, actions, lengths, LongTermOutcome, dones = Sepsis_data[patient_idx]
        lengths = lengths.cpu().numpy().tolist()
        for step in range(lengths-1):
            state, action, reward, next_state, done = (
                states[step], actions[step],
                # Note: 第一层关注患者的关键生理信号，因为需要先保证患者存活；第二层关注患者对肝素的response，用Platelets, aPTT来感知；第三层关注患者长期结局
                # Feature index: {Platelets: 12, aPTT: 16, Lactate: 21, SOFA: 35 / -1}
                # 所有temporal variable已经标准化到了 [-1, 1]
                (states[step + 1][-1] == states[step][-1] and states[step + 1][-1] > -1) * args.C_0 + \
                (states[step + 1][-1] - states[step][-1]) * args.C_1 + \
                F.tanh(states[step + 1][21] - states[step][21]) * args.C_2 + \
                SIT_heparin_response(states[step + 1][12], [-0.86322, -0.55927]) * args.C_3 + \
                SIT_heparin_response(states[step + 1][16], [0.13923, 0.60281]) * args.C_4 + \
                LongTermOutcome[step+1] * args.terminal_coeff,

                states[step + 1], dones[step+1]
            )
            replay_buffer.add(state.unsqueeze(dim=0), action.unsqueeze(dim=0), reward.unsqueeze(dim=0), next_state.unsqueeze(dim=0), done.unsqueeze(dim=0))
    return replay_buffer


def training_DQN(agent, raw_replay_buffer, args):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(agent.parameters(), lr=args.Lr)
    losses = {'train': []}

    agent.train()
    for epoch in range(args.Epochs):
        train_replay_buffer = copy.deepcopy(raw_replay_buffer)
        print('Batch number: {}\n'.format(train_replay_buffer.size() / args.Batch_size))
        with tqdm(total=int(train_replay_buffer.size() / args.Batch_size),
                  desc='Epoch {}/{}'.format(epoch + 1, args.Epochs)) as pbar:
            for i in range(int(train_replay_buffer.size() / args.Batch_size)):
                optimizer.zero_grad()
                b_s, b_a, b_r, b_ns, b_d = train_replay_buffer.sample(args.Batch_size)  # 不放回采样
                transition_dict_for_train = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                q_values, q_targets = agent.update(transition_dict_for_train)
                loss = criterion(q_values, q_targets.detach())
                loss.backward()
                optimizer.step()

                losses['train'].append(loss.item())
                pbar.set_postfix({
                    'Train MSE:': '{:.6f}'.format(loss.item()),
                })
                pbar.update(1)
    return agent, losses


def testing_DQN(agent, test_data, behavior_policy, fqi_model, args):
    agent.eval()
    with torch.no_grad():
        test_replay_buffer = prepare_replay_buffer(tensor_tuple=test_data, args=args)
        b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        states = torch.cat(transition_dict['states'], dim=0)
        Q_estimate, _, agent_policy = agent.take_action(states)
        agent_policy = agent_policy.detach().cpu().numpy()

        evaluation_results = run_evaluate(agent_policy, transition_dict=transition_dict,
                                          behavior_policy=behavior_policy, Q_estimate=None, fqi_model=fqi_model,
                                          args=args)
    return evaluation_results
