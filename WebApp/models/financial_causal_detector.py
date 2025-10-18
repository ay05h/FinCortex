import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class CausalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CausalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(next(self.parameters()).device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        
        return x, attention

class CausalContextLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CausalContextLayer, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        self.scale = self.scale.to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.fc_o(x)
        
        return x

class FinancialCausalDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=768, num_layers=3, 
                 num_heads=8, dropout=0.3, use_glove=False, glove_path=None):
        super(FinancialCausalDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.context_layers = nn.ModuleList([
            CausalContextLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.context_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.cause_start_classifier = nn.Linear(hidden_dim, 1)
        self.cause_end_classifier = nn.Linear(hidden_dim, 1)
        self.effect_start_classifier = nn.Linear(hidden_dim, 1)
        self.effect_end_classifier = nn.Linear(hidden_dim, 1)
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def create_padding_mask(self, x, pad_idx=0):
        return (x != pad_idx).float()
    
    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        seq_len = text.shape[1]
        
        padding_mask = self.create_padding_mask(text).unsqueeze(-1)
        embedded = self.embedding(text)
        
        packed_embedded = pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=seq_len)
        
        x = outputs
        for i in range(len(self.context_layers)):
            context_input = self.context_norms[i](x)
            context_output = self.context_layers[i](context_input, context_input, context_input)
            x = x + context_output
            
            ff_input = self.ff_norms[i](x)
            ff_output = self.ff_layers[i](ff_input)
            x = x + ff_output
        
        cause_start_logits = self.cause_start_classifier(x).squeeze(-1)
        cause_end_logits = self.cause_end_classifier(x).squeeze(-1)
        effect_start_logits = self.effect_start_classifier(x).squeeze(-1)
        effect_end_logits = self.effect_end_classifier(x).squeeze(-1)
        
        padding_mask = padding_mask.squeeze(-1)
        cause_start_logits = cause_start_logits * padding_mask - 1e10 * (1 - padding_mask)
        cause_end_logits = cause_end_logits * padding_mask - 1e10 * (1 - padding_mask)
        effect_start_logits = effect_start_logits * padding_mask - 1e10 * (1 - padding_mask)
        effect_end_logits = effect_end_logits * padding_mask - 1e10 * (1 - padding_mask)
        
        sentence_rep = torch.mean(x, dim=1)
        
        cause_probs = torch.softmax(cause_start_logits, dim=-1).unsqueeze(-1)
        effect_probs = torch.softmax(effect_start_logits, dim=-1).unsqueeze(-1)
        
        cause_rep = torch.sum(x * cause_probs, dim=1)
        effect_rep = torch.sum(x * effect_probs, dim=1)
        
        relation_input = torch.cat([cause_rep, effect_rep], dim=-1)
        relation_logits = self.relation_classifier(relation_input)
        
        return {
            "cause_start_logits": cause_start_logits,
            "cause_end_logits": cause_end_logits,
            "effect_start_logits": effect_start_logits,
            "effect_end_logits": effect_end_logits,
            "relation_logits": relation_logits
        }