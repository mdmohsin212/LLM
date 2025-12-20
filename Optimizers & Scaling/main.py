import torch
import math
import torch.optim as optim

CONFIG = {
    'vocab_size': 4096,
    'embed_size': 256,
    'layers': 4,
    'heads': 4,
    'batch_size': 32,
    'block_size': 128,
    'max_iters': 5000,
    'learning_rate': 1e-4
}

# ================================
# 2. Lion Optimizer
# ================================
class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Main update loop
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Init momentum buffer
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Apply decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Compute sign-based update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Update momentum buffer
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

# ================================
# 3. LR Scheduler (Warmup + Cosine)
# ================================
def get_lr(step, config):
    warmup_iters = 200
    lr_decay_iters = config['max_iters']
    min_lr = config['learning_rate'] / 10
    max_lr = config['learning_rate']

    # Linear warmup
    if step < warmup_iters:
        return max_lr * (step + 1) / (warmup_iters + 1)

    # Final constant LR
    if step > lr_decay_iters:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ================================
# 4. Training Setup
# ================================

# Example placeholder model (replace with your own)
# model = TransformerLM(...)
# model.to(DEVICE)

optimizer_type = "Lion"  # Choose: "AdamW", "Lion", "RMSProp"

# Choose optimizer
if optimizer_type == "Lion":
    optimizer = Lion(model.parameters(), lr=CONFIG['learning_rate']/3, weight_decay=1e-2)
    print("Using Lion Optimizer")
elif optimizer_type == "RMSProp":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=CONFIG['learning_rate'], alpha=0.99)
    print("Using RMSProp Optimizer")
else:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=1e-1
    )
    print("Using AdamW Optimizer")

print("Starting Training...")

for step in range(CONFIG['max_iters']):
    # Update learning rate
    lr = get_lr(step, CONFIG)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Load training batch
    xb, yb = get_batch()

    # Forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Parameter update
    optimizer.step()

    # Log every 100 steps
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

print("Training Complete!")