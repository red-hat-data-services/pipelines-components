from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


class TmpEventHandler(BaseEventHandler):
    """Exists temporarily only for the purpose of satisying type hinting checks"""

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        pass

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        pass
