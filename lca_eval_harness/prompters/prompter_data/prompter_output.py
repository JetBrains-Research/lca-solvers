from dataclasses import dataclass


@dataclass
class PrompterOutput:
    prompt: str | None = None
    target: str | None = None
