class ApiError(Exception):
    """An error that maps to a JSON HTTP response: {"detail": {"code", "message", ...}}."""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None, **extra):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}
        self.extra = extra
