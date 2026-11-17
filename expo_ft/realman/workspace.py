"""Typed rejection of an unexecuted target, distinct from hardware faults."""


class WorkspaceMotionRejected(RuntimeError):
    """Current pose is valid, but the requested joint path leaves the workspace."""
