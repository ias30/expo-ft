"""Controller gripper convention: 0=closed, 1=open."""
OPEN_THRESHOLD = 0.7
CLOSE_THRESHOLD = 0.3


def gripper_target(value, current_state):
    return int(value >= OPEN_THRESHOLD) if current_state == 0 else int(value > CLOSE_THRESHOLD)
