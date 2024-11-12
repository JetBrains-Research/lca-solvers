from dataclasses import dataclass

from pydriller import ModifiedFile, ModificationType


@dataclass
class FileModStats:
    char_len: int
    line_len: int


@dataclass
class FileModMetadata:
    mod_type: str
    old_path: str | None
    new_path: str | None
    path: str | None
    content: str | None
    _content_before: str | None
    _content_after: str | None
    stats_before: FileModStats | None = None
    stats_after: FileModStats | None = None

    def __post_init__(self):
        if self._content_before is not None:
            self.stats_before = FileModStats(
                char_len=len(self._content_before),
                line_len=len(self._content_before.split('\n')),
            )
            self._content_before = None

        if self._content_after is  not None:
            self.stats_after = FileModStats(
                char_len=len(self._content_after),
                line_len=len(self._content_after.split('\n')),
            )
            self._content_after = None

    @classmethod
    def from_modified_file(cls, mod: ModifiedFile) -> 'FileModMetadata':
        (content, path, content_before, content_after,) = (None, None, None, None,)
        if mod.change_type == ModificationType.ADD:
            content = mod.source_code
            content_after = mod.source_code
            path = mod.new_path
        elif mod.change_type == ModificationType.DELETE:
            content = mod.source_code_before
            content_before = mod.source_code_before
            path = mod.old_path
        elif mod.change_type == ModificationType.MODIFY:
            content = mod.source_code
            content_before = mod.source_code_before
            content_after = mod.source_code
            path = mod.new_path

        file_mod_data = {
            "mod_type": mod.change_type.name,
            "old_path": mod.old_path,
            "new_path": mod.new_path,
            "path": path,
            "content": content,
            "_content_before": content_before,
            "_content_after": content_after,
        }
        return cls(**file_mod_data)

    @classmethod
    def from_json(cls, data: dict) -> 'FileModMetadata':
        if data['stats_before'] is not None:
            data['stats_before'] = FileModStats(**data['stats_before'])
        if data['stats_after'] is not None:
            data['stats_after'] = FileModStats(**data['stats_after'])
        return cls(**data)
