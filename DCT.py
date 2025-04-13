class DCT(nn.Module):
    def __init__(self, resolution, device, norm=None, bias=False):
        super(DCT, self).__init__()
        self.resolution = resolution
        self.norm = norm
        self.device = device

        I = torch.eye(self.resolution, device=self.device)
        self.forward_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.forward_transform.weight.data = self._dct(I, norm=self.norm).data.t()
        self.forward_transform.weight.requires_grad = False

        self.inverse_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.inverse_transform.weight.data = self._idct(I, norm=self.norm).data.t()
        self.inverse_transform.weight.requires_grad = False

    def _dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def _idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape)

    def forward(self, x):
        B, S, D = x.shape  # B: batch_size, S: num_patches, D: feature_size

        # Flatten to [B * S, D]
        x = x.reshape(-1, D)  # 使用 reshape 代替 view
        print(f"Shape of x forward before transformation: {x.shape}")
        # Apply forward transformation (DCT)
        X1 = self.forward_transform(x)
        print(f"Shape of X1 forward: {X1.shape}")
        # Apply the DCT to the transformed tensor
        X2 = self.forward_transform(X1)  # Apply linear transformation
        print(f"Shape of X2 forward after transpose and transformation: {X2.shape}")
        # Reshape back to original shape [B, S, D]
        return X2.reshape(B, S, D)  # 使用 reshape 代替 view

    def inverse(self, x):
        B, S, D = x.shape  # B: batch_size, S: num_patches, D: feature_size

        # Flatten to [B * S, D]
        x = x.reshape(-1, D)  # Flatten the input tensor
        print(f"Shape of x before transformation: {x.shape}")

        # Apply inverse transformation (Inverse DCT)
        X1 = self.inverse_transform(x)  # [B * S, D]
        print(f"Shape of X1: {X1.shape}")

        # Ensure X1 is in the correct shape for the next operation
        X1 = X1.reshape(B, S, D)  # [B, S, D] after reshape
        print(f"Shape of X1 after reshape: {X1.shape}")

        # Ensure X1 is transposed correctly for the inverse operation
        X2 = self.inverse_transform(X1.transpose(-2, -3))  # Transpose last two dimensions
        print(f"Shape of X2 after transpose and transformation: {X2.shape}")

        # Return to original shape [B, S, D]
        return X2.transpose(-1, -2).reshape(B, S, D)


class DCTDistillationModule(nn.Module):
    def __init__(self, resolution, device, norm=None):
        super(DCTDistillationModule, self).__init__()
        self.dct = DCT(resolution, device, norm)

    def forward(self, pseudo_teacher_features, teacher_features):
        """
        Args:
            pseudo_teacher_features (torch.Tensor): Shape [B, S, 1024], generated pseudo-teacher features
            teacher_features (torch.Tensor): Shape [B, S, 1024], teacher features
        """
        # Apply DCT to both pseudo-teacher features and teacher features
        dct_pseudo_teacher = self.dct.forward(pseudo_teacher_features)  # Apply DCT
        dct_teacher = self.dct.forward(teacher_features)  # Apply DCT

        print("Shape of pseudo_teacher_features:", pseudo_teacher_features.shape)
        print("Shape of teacher_features:", teacher_features.shape)


        # Apply inverse DCT to both transformed features
        idct_pseudo_teacher = self.dct.inverse(dct_pseudo_teacher)  # Inverse DCT
        idct_teacher = self.dct.inverse(dct_teacher)  # Inverse DCT



        return idct_pseudo_teacher, idct_teacher












class DCTDistillationLoss(nn.Module):
    def __init__(self, dct_module, distillation_weight=1.0, dct_weight=1.0):
        """
        Args:
            distillation_weight (float): 权重用于控制伪教师特征图与教师特征图之间的蒸馏损失
            dct_weight (float): 权重用于控制经过DCT与逆DCT的蒸馏损失
        """
        super(DCTDistillationLoss, self).__init__()
        self.dct_module = dct_module
        self.distillation_weight = distillation_weight
        self.dct_weight = dct_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pseudo_teacher_features, teacher_features):
        """
        Args:
            pseudo_teacher_features (torch.Tensor): 生成的伪教师特征图，Shape [B, S, 1024]
            teacher_features (torch.Tensor): 教师特征图，Shape [B, S, 1024]
        """
        # 计算常规蒸馏损失
        distillation_loss = self.mse_loss(pseudo_teacher_features, teacher_features)

        # 计算通过DCT与逆DCT后的蒸馏损失
        idct_pseudo_teacher, idct_teacher = self.dct_module(pseudo_teacher_features, teacher_features)
        dct_loss = self.mse_loss(idct_pseudo_teacher, idct_teacher)

        # 总损失 = 常规蒸馏损失 + DCT蒸馏损失
        total_loss = self.distillation_weight * distillation_loss + self.dct_weight * dct_loss

        return total_loss
