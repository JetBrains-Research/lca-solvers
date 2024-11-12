from dataclasses import dataclass

@dataclass
class ClassifiedLine:
    lineNumber: int
    contents: str
    lineTypes: list[str]


@dataclass
class ProcessedFile:
    filename: str
    lines: list[ClassifiedLine]

    @classmethod
    def from_json(cls, json_data: dict) -> 'ProcessedFile':
        lines = list()
        for json_line in json_data['lines']:
            lines.append(ClassifiedLine(**json_line))

        return cls(
            filename=json_data['filename'],
            lines=lines,
        )
