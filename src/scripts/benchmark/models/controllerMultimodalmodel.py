import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, search_space, hidden_size=64):
        super().__init__()
        self.search_space = search_space
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.param_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, len(choices))
            for name, choices in search_space.items()
        })
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def sample_config(self):
        device = next(self.parameters()).device
        h = torch.zeros(1, 1, self.hidden_size, device=device)
        c = torch.zeros(1, 1, self.hidden_size, device=device)
        
        out, (h, c) = self.lstm(self.start_token, (h, c))
        config = {}
        log_probs = []

        for name, head in self.param_heads.items():
            logits = head(out[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            idx = dist.sample()
            config[name] = self.search_space[name][idx.item()]
            log_probs.append(dist.log_prob(idx))
            
            # MELHORIA: Usar a escolha como entrada para o próximo passo do LSTM (melhora a dependência)
            # Embedding simples para a escolha do índice
            # embedding = self.embeddings[name](idx.unsqueeze(0))
            # out, (h,c) = self.lstm(embedding.unsqueeze(1), (h, c))
            # (Nota: Para simplificar, mantive a versão original, mas esta seria uma melhoria avançada)

        return config, torch.stack(log_probs).sum()
