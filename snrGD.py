import torch

class SNRGatedDescent(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-1, beta=0.999, eps=1e-8, m=1, use_param=True, log_omega=False):
        defaults = dict(lr=lr, beta=beta, eps=eps, m=m, use_param = use_param, log_omega=log_omega)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_count += 1

        for group in self.param_groups:

            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            m = group["m"]
            use_param = group["use_param"]
            #mom = group["mom"]
            #wd = group["weight_decay"]

            params = []
            grads = []
            m_rho = []
            v_rho = []
            #prev = [] if mom !=0.0 else None

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if not state:
                    state["m_rho"] = torch.zeros_like(p)
                    state["v_rho"] = torch.zeros_like(p)
                    #state["prev"] = torch.zeros_like(p)

                params.append(p)
                grads.append(p.grad)
                m_rho.append(state["m_rho"])
                v_rho.append(state["v_rho"])
                #if mom != 0.0:
                    #prev.append(state["prev"])

            if len(params) == 0:
                continue

            # rho definition (ablation: theta * g vs g)
            if use_param:
                rho = torch._foreach_mul(params, grads)   # theta * g
            else:
                rho = grads                               # g only

            # m_rho update
            torch._foreach_mul_(m_rho, beta)
            torch._foreach_add_(m_rho, rho, alpha=(1 - beta))

            # v_rho update
            rho2 = torch._foreach_mul(rho, rho)
            torch._foreach_mul_(v_rho, beta)
            torch._foreach_add_(v_rho, rho2, alpha=(1 - beta))

            # SNR omega = m_rho/(v_rho.sqrt() + eps)
            sqrt_v_rho = torch._foreach_sqrt(v_rho)
            torch._foreach_add_(sqrt_v_rho, eps)

            abs_m_rho = torch._foreach_abs(m_rho)
            snr = torch._foreach_div(abs_m_rho, sqrt_v_rho)

            # bias correction sqrt
            bias_sqrt = (1 - beta ** self.step_count) ** -0.5
            torch._foreach_mul_(snr, bias_sqrt)

            # omega^m
            snr_pow = torch._foreach_pow(snr, m)
            
            # Omega = omega^m / (1 + omega^m)
            denom = torch._foreach_add(snr_pow, 1.0)
            Omega = torch._foreach_div(snr_pow, denom)
            # --- Store Omega samples (avoid memory blowup) ---
            if group.get("log_omega", False):
                omega_samples = []

                for o in Omega:
                    flat = o.flatten()
                    if flat.numel() > 1000:
                        idx = torch.randperm(flat.numel(), device=flat.device)[:1000]
                        flat = flat[idx]
                    omega_samples.append(flat.detach().cpu())

                self._last_omega_samples = torch.cat(omega_samples)

            # Omega * grad
            scaled_grad = torch._foreach_mul(Omega, grads)

            #if wd != 0.0:
                 #torch._foreach_add_(params, params, alpha=-lr * wd)
                        
            # momentum smoothing
            #if mom != 0.0:
                #torch._foreach_mul_(prev, mom)
                #torch._foreach_add_(prev, scaled_grad, alpha=(1-mom))
                #torch._foreach_add_(params, prev, alpha=-lr)
            #elif mom == 0.0:
            torch._foreach_add_(params, scaled_grad, alpha=-lr)
