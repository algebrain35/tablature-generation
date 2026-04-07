import torch.nn as nn
import torch.functionalm as F
class FretboardEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        
        self.string_embed = nn.Embedding(7, embed_dim // 4)
        self.fret_embed   = nn.Embedding(22, embed_dim // 4)
        self.pitch_embed  = nn.Embedding(128, embed_dim // 2)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, string, fret, pitch):
        s = self.string_embed(string)
        f = self.fret_embed(fret)
        p = self.pitch_embed(pitch)
        
        x = torch.cat([s, f, p], dim=-1)
        return self.norm(self.proj(x))
class FretboardDecoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.norm   = nn.LayerNorm(embed_dim)
        self.proj   = nn.Linear(embed_dim, 126)
    
    def forward(self, hidden):
        x      = self.norm(hidden)
        logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)

class FretboardNextToken(nn.Module):
    def __init__(self, window=8, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.window   = window
        self.encoder  = FretboardEncoder(embed_dim)
        self.mlp      = nn.Sequential(
            nn.Linear(window * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 126)
        )
    
    def forward(self, context_positions):
        embedded = self.encoder(context_positions)
        flat     = embedded.view(embedded.size(0), -1)
        return F.log_softmax(self.mlp(flat), dim=-1) 
