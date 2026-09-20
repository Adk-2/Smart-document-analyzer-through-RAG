import threading


class WorkspaceManager:

    _local = threading.local()
    _default = "default_workspace"

    @classmethod
    def set_workspace(cls, name):

        clean = (name or "").strip()

        if not clean:
            raise ValueError("Workspace name must be set before use.")

        cls._local.workspace = clean

    @classmethod
    def get_workspace(cls):

        return getattr(
            cls._local,
            "workspace",
            cls._default
        )
