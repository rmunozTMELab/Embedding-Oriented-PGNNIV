import torch

def compute_fourier_base(num_modes, X_mesh, Y_mesh):

    base = []
    mode_idx = 0
    max_order = 2 * int(torch.sqrt(torch.tensor(num_modes))) + 1

    max_m = max_n = int(torch.sqrt(torch.tensor(num_modes))) + 1

    for s in range(max_order):  # s = m + n
        for m in range(s + 1):
            n = s - m

            if mode_idx == 0:
                base.append(torch.ones_like(X_mesh))
                mode_idx += 1
                if mode_idx >= num_modes:
                    break

            if m > 0 and n > 0:
                # A_mn
                base.append(torch.cos(2 * torch.pi * m * X_mesh / (max_m - 1) + 2 * torch.pi * n * Y_mesh / (max_n - 1)))
                mode_idx += 1
                if mode_idx >= num_modes:
                    break

                # B_mn
                base.append(torch.sin(2 * torch.pi * m * X_mesh / (max_m - 1) + 2 * torch.pi * n * Y_mesh / (max_n - 1)))
                mode_idx += 1
                if mode_idx >= num_modes:
                    break

        if mode_idx >= num_modes:
            break

    base = torch.stack(base, dim=0)

    return base