"""Simple default gridder as fallback."""


class DefaultGridder:
    @classmethod
    def from_pyvisgen(cls):
        raise NotImplementedError(
            "The default gridder will be added in a future release."
        )
