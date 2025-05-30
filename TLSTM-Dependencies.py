import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, timestamps):
        b, seq, _ = inputs.size()
        device_ = inputs.device
        h = torch.zeros(b, self.hidden_size, device=device_)
        c = torch.zeros(b, self.hidden_size, device=device_)

        outputs = []
        for s in range(seq):
            if s == 0:
                decay = torch.ones(b, 1, device=device_)
            else:
                delta_t = (timestamps[:, s] - timestamps[:, s-1]).clamp(min=1e-6)
                decay = 1 / torch.log(torch.exp(torch.ones(1, device=device_)) + delta_t)
                decay = decay.unsqueeze(1)

            c_short = torch.tanh(self.W_d(c))
            c_long = c - c_short
            c_short_discounted = c_short * decay
            c_adj = c_long + c_short_discounted

            outs = self.W_all(h) + self.U_all(inputs[:, s, :])
            f, i, o, c_tmp = torch.chunk(outs, 4, dim=1)
            f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
            c_tmp = torch.tanh(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_size]
        return outputs, (h, c)

class CoxPHAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CoxPHAttentionModel, self).__init__()
        self.lstm = TimeLSTM(input_size, hidden_size)
        self.attention_fc = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        self.output_fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs, timestamps):
        lstm_out, _ = self.lstm(inputs, timestamps)          # [batch, seq_len, hidden_size]
        u = torch.tanh(self.attention_fc(lstm_out))          # [batch, seq_len, hidden_size]
        attention_weights = F.softmax(self.context_vector(u), dim=1)  # [batch, seq_len, 1]
        V = torch.sum(attention_weights * lstm_out, dim=1)   # [batch, hidden_size]
        output = self.output_fc(V)                           # [batch, 1]
        output = output.squeeze(1)                           # [batch]
        return output, attention_weights

class SurvivalModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SurvivalModel, self).__init__()
        self.model = CoxPHAttentionModel(input_size, hidden_size)

    def forward(self, features, timestamps):
        """
        features: [batch, seq_len, input_size]
        timestamps: [batch, seq_len]
        """
        risk, attention_weights = self.model(features, timestamps)
        return risk, attention_weights


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, logits, durations, events):
        """
        logits: Risk scores from the model [batch]
        durations: Survival times [batch]
        events: Event indicators [batch] (1=event, 0=censored)
        """
        sorted_indices = torch.argsort(durations, descending=True)
        sorted_logits = logits[sorted_indices]
        sorted_events = events[sorted_indices]

        exp_logits = torch.exp(sorted_logits)
        log_cumsum = torch.log(torch.cumsum(exp_logits, dim=0))
        loss = -(sorted_logits - log_cumsum) * sorted_events
        return loss.sum()