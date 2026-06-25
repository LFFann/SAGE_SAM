"""Scalar schedules for SAGE-SAM training."""

def linear_decay(step: int, start: float, end: float, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    ratio = min(1.0, max(0.0, step / float(total_steps)))
    return start + (end - start) * ratio
