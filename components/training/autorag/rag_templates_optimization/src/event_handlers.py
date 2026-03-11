from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


class TmpEventHandler(BaseEventHandler):
    """Exists temporarily only for the purpose of satisfying type hinting checks."""

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        """No-op for status changes."""
        pass

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        """No-op for pattern creation."""
        pass
