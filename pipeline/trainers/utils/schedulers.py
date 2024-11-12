import math


def get_lr_from_cosine_scheduler_with_linear_warmup(iter_num: int,
                                                    min_lr: float,
                                                    max_lr: float,
                                                    warmup_iters: int,
                                                    lr_decay_iters: int,
                                                    ) -> float:
    if iter_num < warmup_iters:  # warmup
        return max_lr * (iter_num + 1) / warmup_iters
    elif iter_num > lr_decay_iters:  # constant lr
        return min_lr
    else:  # cosine wave
        decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
        return min_lr + (max_lr - min_lr) / 2 * (1 + math.cos(math.pi * decay_ratio))
