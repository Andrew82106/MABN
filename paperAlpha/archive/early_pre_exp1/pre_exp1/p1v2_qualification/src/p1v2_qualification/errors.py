class QualificationError(Exception):
    """A structured, safe-to-report qualification failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class PathPolicyError(QualificationError):
    pass


class ContractError(QualificationError):
    pass


class FakeProviderError(QualificationError):
    pass


class FakeProviderTimeout(QualificationError):
    pass
