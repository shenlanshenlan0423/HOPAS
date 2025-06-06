# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：seq2seq.py
@IDE ：PyCharm
"""
from definitions import *


class TimeEmbedding:
    def __init__(self, d):
        all_tau = torch.arange(1, horizon + 1, dtype=torch.float)
        time_embedding = []
        for tau in all_tau:
            sin_value = torch.stack([torch.sin(tau * j / (horizon * d)) for j in range(d)])
            cos_value = torch.stack([torch.cos(tau * j / (horizon * d)) for j in range(d)])
            time_embedding.append(torch.cat([sin_value, cos_value]))
        self.d = d
        self.time_embedding = torch.stack(time_embedding).to(device)


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'concat2':
            self.attn = nn.Linear(hidden_size * 3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'self':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=False),
                nn.Tanh()
            )

    def dot_score(self, hidden, encoder_output):
        # 点积注意力机制,计算隐藏状态 hidden 和编码输出 encoder_output 的点积
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        # 一般注意力机制,先对 encoder_output 进行仿射变换,然后与 hidden 计算点积
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)  # torch.Size([batch_size, sen_len])

    def concat_score(self, hidden, encoder_output):
        # 拼接注意力机制,将 hidden 和 encoder_output 拼接后经过一个线性层,再与权重向量 self.v 计算点积
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def concat_score2(self, hidden, encoder_output):
        # 另一种拼接注意力机制,除了拼接 hidden 和 encoder_output 外,还拼接了它们的逐元素乘积
        h = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        h = torch.cat((h, hidden * encoder_output), 2)
        energy = self.attn(h).tanh()
        return torch.sum(self.v * energy, dim=2)

    def self_score(self, encoder_output):
        # 自注意力机制,先使用一个线性层对 encoder_output 进行变换,然后与 encoder_output 计算点积
        return self.attn(encoder_output)  # torch.Size([batch_size, sen_len, 1])

    def forward(self, hidden, encoder_outputs, mask):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'concat2':
            attn_energies = self.concat_score2(hidden, encoder_outputs)
        elif self.method == 'self':
            # attn_energies = self.self_score(hidden, encoder_outputs)
            attn_energies = self.self_score(encoder_outputs).squeeze(-1)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        attn_energies = torch.exp(attn_energies)
        attn_energies = attn_energies * mask.float().to(device)  # Note: 带时间掩码的自注意力
        attn_energies_sum = attn_energies.sum(dim=1)
        attn_energies = attn_energies / (attn_energies_sum.unsqueeze(1) + 0.000001)
        return attn_energies


class AttentionEncoder(nn.Module):
    def __init__(self, static_dim, temporal_dim, d, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.d = d
        self.time_embedding = TimeEmbedding(d).time_embedding
        self.static_dim = static_dim
        self.static_embedding = nn.Linear(static_dim, embedding_dim)
        self.temporal_embedding = nn.Linear(temporal_dim, embedding_dim)
        self.action_embedding = nn.Linear(action_dim, embedding_dim)

        self.attn = Attention('self', hidden_size)
        self.LSTM = nn.LSTM(
            input_size=action_dim + embedding_dim * 2 + d * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, a, time_mask):
        temporal_e = self.dropout(self.temporal_embedding(s[:, :, self.static_dim:]))
        static_e = self.static_embedding(s[:, :, :self.static_dim])
        time_e = self.time_embedding.unsqueeze(0).expand(s.shape[0], horizon, 2 * self.d)
        # action_e = self.action_embedding(a)
        # outputs 包含了 LSTM 在每个时间步上的隐藏状态输出; h_n 表示 LSTM 在最后一个时间步的隐藏状态输出;
        # outputs[:, -1, :].unsqueeze(0) == h_n: True
        outputs, (h_n, c_n) = self.LSTM(torch.cat([a, static_e, temporal_e, time_e], dim=-1))

        attn_weights = self.attn.forward(None, outputs, time_mask)  # Self Attention
        attn_represent = torch.sum(outputs * attn_weights.unsqueeze(2), dim=1)  # Patient-level representation
        return static_e, attn_represent, outputs, h_n, c_n


class AttentionDecoder(nn.Module):
    def __init__(self, temporal_dim, d, embedding_dim, latent_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.d = d
        self.time_embedding = TimeEmbedding(d).time_embedding
        self.temporal_embedding = nn.Linear(temporal_dim, embedding_dim)
        self.action_embedding = nn.Linear(action_dim, embedding_dim)
        self.LSTM = nn.LSTM(
            input_size=hidden_size + embedding_dim * 2 + d * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.attn = Attention('general', hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pred_next_s = nn.Sequential(
            nn.Linear(hidden_size * 2 + action_dim, temporal_dim),
            nn.Tanh()  # Output range (-1, 1)
        )

    def forward(self, attn_represent, t, input_t, a_t, static_e_t, encoder_outputs, hidden, cell, time_mask):
        time_e = self.time_embedding.unsqueeze(0).expand(attn_represent.shape[0], horizon, 2 * self.d)[:, t, :]
        input_t_e = self.dropout(self.temporal_embedding(input_t))
        condition = torch.cat([static_e_t, time_e], dim=-1)
        output, (hidden, cell) = self.LSTM(torch.cat([attn_represent, input_t_e, condition], dim=-1).unsqueeze(1), (hidden, cell))
        # torch.cat([z, input_t, condition], dim=-1): (batch_size, output_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)

        # Decoder当前时刻的output和encoder所有时刻的output计算注意力得分
        attn_weights = self.attn.forward(output, encoder_outputs, time_mask)
        context = torch.sum(encoder_outputs * attn_weights.unsqueeze(2), dim=1)  # 根据注意力权重更新context vector

        concat_vector = torch.cat([context, output.squeeze(1), a_t], dim=-1)
        pred_next_s = self.pred_next_s(concat_vector)
        return torch.zeros((pred_next_s.shape[0], 1)), pred_next_s, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, static_dim, hidden_size, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, action, trg, mask, teacher_forcing_ratio):
        """
        :param src: Source trajectory
        :param action: Discretized heparin
        :param trg: Target trajectory
        :param mask: Time mask
        :param teacher_forcing_ratio: 在training阶段，以一定概率采用ground truth辅助训练
        :return:
        """
        # Encoder 此处的hidden, cell是encoder最后一个时刻的
        static_e, attn_represent, encoder_outputs, hidden, cell = self.encoder(src, action, mask)

        batch_size = src.shape[0]
        src_len, trg_len = horizon, horizon
        trg_vocab_size = len(temporal_cols)
        # Tensor for storing decoder outputs
        pred_s_ys = torch.zeros(batch_size, src_len).to(device)
        pred_next_ss = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        # 第0时刻，decoder接收到上一个时刻的输出为全0
        input_t = torch.zeros((batch_size, trg_vocab_size)).to(device)

        # Teacher-forcing technique
        for t in range(trg_len):
            action_t = action[:, t, :].squeeze(1)
            static_e_t = static_e[:, t, :].squeeze(1)
            # Decoder  encoder_outputs用于注意力机制; 上一时刻的hidden, cell作为下一时刻输入的一部分
            pred_s_y, pred_next_s, hidden, cell = self.decoder(attn_represent, t, input_t, action_t, static_e_t, encoder_outputs, hidden, cell, mask)
            # place predictions in a tensor holding predictions for each token
            pred_s_ys[:, t] = pred_s_y.squeeze(-1)
            pred_next_ss[:, t, :] = pred_next_s
            # decide if we are going to use teacher-forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # if teacher forcing, use actual next token as next input; if not, use predicted token
            input_t = trg[:, t, -trg_vocab_size:] if teacher_force else pred_next_s
        return pred_s_ys, pred_next_ss


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    在前 num_warmup_steps 步内,学习率线性增加从 0 到初始学习率; 在剩余 num_training_steps - num_warmup_steps 步内,学习率线性衰减到 0
    预热阶段: 在训练开始时,学习率从 0 逐步增加到初始学习率,可以帮助模型更好地收敛。这在训练大型模型时很有用。
    线性衰减: 在训练后期,学习率线性衰减到 0,可以帮助模型更好地稳定和收敛。
    灵活性: 可以通过调整 num_warmup_steps 和 num_training_steps 来控制预热阶段和衰减阶段的长度,以适应不同的训练需求。
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(model, train_loader, valid_loader, model_name, fold_label, args):
    print("Start training, total parameter:{}, trainable:{}\n".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))

    t_batch = len(train_loader)
    v_batch = len(valid_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    training_steps = args.epochs * t_batch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, training_steps)

    train_losses, val_losses = [], []
    best_mse, best_auc = float('inf'), -float('inf')
    for epoch in range(args.epochs):
        with tqdm(total=1, desc='Epoch {}/{}'.format(epoch + 1, args.epochs)) as pbar:
            model.train()
            train_loss = 0
            for (s, a, ns, lens, outcomes) in train_loader:
                optimizer.zero_grad()
                state_mask = torch.zeros_like(ns[:, :, len(static_cols):], dtype=torch.bool)
                time_mask = torch.zeros((ns.shape[0], horizon), dtype=torch.bool)
                for i in range(state_mask.shape[0]):
                    state_mask[i, :lens[i], :] = True
                    time_mask[i, :lens[i]] = True
                # Note: time mask用于计算掩码注意力
                _, pred_s = model(src=s, action=a, trg=s, mask=time_mask,
                                  teacher_forcing_ratio=args.teacher_forcing_ratio)
                loss = F.mse_loss(pred_s[state_mask], s[:, :, len(static_cols):][state_mask])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)  # 如果梯度的 L2 范数超过 5,就会将梯度缩放到范数为 5
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_losses.append(train_loss / t_batch)

            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_mse, val_true_y, val_pred_y = [], [], []
                for (s, a, ns, lens, outcomes) in valid_loader:
                    state_mask = torch.zeros_like(ns[:, :, len(static_cols):], dtype=torch.bool)
                    time_mask = torch.zeros((ns.shape[0], horizon), dtype=torch.bool)
                    for i in range(state_mask.shape[0]):
                        state_mask[i, :lens[i], :] = True
                        time_mask[i, :lens[i]] = True
                    _, pred_s = model(src=s, action=a, trg=s, mask=time_mask, teacher_forcing_ratio=0)
                    loss = F.mse_loss(pred_s[state_mask], s[:, :, len(static_cols):][state_mask])
                    val_mse.append(loss.item())
                    val_loss += loss.item()
                val_losses.append(val_loss / v_batch)

            # assert not math.isnan(np.mean(val_mse)), "Gradient vanish..."
            if math.isnan(np.mean(val_mse)):
                break
            if np.mean(val_mse) < best_mse and not math.isnan(np.mean(val_mse)):
                best_mse = np.mean(val_mse)
                save_flag = 'Save model with mse: {:.6f}'.format(best_mse)
                torch.save(model, MODELS_DIR + '/{}-{}.pt'.format(model_name, fold_label))
            else:
                save_flag = 'Waiting...'

            pbar.set_postfix({
                'Train loss:': '{:.6f}'.format(train_loss / t_batch),
                'Val loss:': '{:.6f}'.format(val_loss / v_batch),
                'Val MSE:': '{:.6f}'.format(np.mean(val_mse)),
                'Model Save': '{}'.format(save_flag)
            })
            pbar.update(1)
    # return {'train loss': train_losses, 'val loss': val_losses}


def test(model, test_loader):
    model.eval()
    masked_s, masked_pred_s = [], []
    all_s, all_pred_s = [], []
    masked_true_s_y, masked_pred_s_y = [], []
    for (s, a, ns, lens, outcomes) in test_loader:
        time_mask = torch.zeros((ns.shape[0], horizon), dtype=torch.bool)
        for i in range(s.shape[0]):
            time_mask[i, :lens[i]] = True
        # inference不能使用ground truth
        _, pred_s = model(src=s, action=a, trg=s, mask=time_mask, teacher_forcing_ratio=0)
        # masked_true_s_y.extend(outcomes[time_mask].detach().cpu().numpy().tolist())
        for i in range(s.shape[0]):
            all_s.append(s[i, :, :].detach().cpu().numpy())
            all_pred_s.append(pred_s[i, :, :].detach().cpu().numpy())
            masked_s.append(s[i, :lens[i], :].detach().cpu().numpy())
            masked_pred_s.append(pred_s[i, :lens[i], :].detach().cpu().numpy())
    return all_s, all_pred_s, masked_s, masked_pred_s, masked_true_s_y, masked_pred_s_y
