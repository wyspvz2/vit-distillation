class FC_Classifier(nn.Module):
    def __init__(self, n_classes, input_dim=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim) if input_dim else nn.Identity(),
            nn.Linear(input_dim, 512) if input_dim else nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

    def forward(self, X):
        return self.block(X)

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, dim=768, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):  # x: [B, C, H, W] or [B, N, D]
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)  # -> [B, N, C]
        return self.transformer(x).mean(dim=1)  # -> [B, D]

class CMKD(nn.Module):
    def __init__(self, input_channel_first=768, input_channel_second=768, num_classes=10):
        super(CMKD, self).__init__()

        self.first_enc_inv = SimpleTransformerEncoder(dim=input_channel_first)
        self.first_enc_spec = SimpleTransformerEncoder(dim=input_channel_first)
        self.second_enc_inv = SimpleTransformerEncoder(dim=input_channel_second)
        self.second_enc_spec = SimpleTransformerEncoder(dim=input_channel_second)

        self.task_dom = FC_Classifier(2)
        self.task_dom2 = FC_Classifier(2)
        self.task_cl = nn.LazyLinear(num_classes)
        self.task_cl2 = nn.LazyLinear(num_classes)
        self.task_cl3 = FC_Classifier(num_classes)
        self.discr = FC_Classifier(2)

        self.projF = ProjHead(256)
        self.projS = ProjHead(256)
    def forward(self, x, lambda_val=1.):
        fx, sx = x  

        f_emb_inv = self.first_enc_inv(fx)
        f_emb_spec = self.first_enc_spec(fx)
        s_emb_inv = self.second_enc_inv(sx)
        s_emb_spec = self.second_enc_spec(sx)

        nfeat = f_emb_inv.shape[1] // 2

        f_shared_discr = self.projF(f_emb_inv)
        s_shared_discr = self.projS(s_emb_inv)

        f_domain_discr = f_emb_spec[:, :nfeat]
        f_domain_useless = f_emb_spec[:, nfeat:]
        s_domain_discr = s_emb_spec[:, :nfeat]
        s_domain_useless = s_emb_spec[:, nfeat:]

        f_task_feat = torch.cat([f_shared_discr, f_domain_discr], dim=1)
        s_task_feat = torch.cat([s_shared_discr, s_domain_discr], dim=1)

        pred_f_emb_dom = torch.cat([
            self.task_dom(f_domain_discr),
            self.task_dom2(f_domain_useless)
        ], dim=0)

        pred_s_emb_dom = torch.cat([
            self.task_dom(s_domain_discr),
            self.task_dom2(s_domain_useless)
        ], dim=0)

        return f_shared_discr, s_shared_discr, \
              f_domain_discr, f_domain_useless, s_domain_discr, s_domain_useless, \
              pred_f_emb_dom, pred_s_emb_dom, \
              self.task_cl(f_task_feat), self.task_cl2(s_task_feat), \
              self.discr(grad_reverse(f_shared_discr, lambda_val)), self.discr(grad_reverse(s_shared_discr, lambda_val)), \
              self.task_cl3(f_shared_discr), self.task_cl3(s_shared_discr), \
              self.task_cl3(f_domain_discr), self.task_cl3(s_domain_discr)


      
