import dataclasses


@dataclasses.dataclass
class DatapointBase:
    repo: str
    commit: str
    relevant_extensions: list[str]
    context_dict: dict[str, str] | None = None
    completion_dict: dict[str, str] | None = None
    context: str | None = None
    completion: str | None = None
    context_len: int | None = None

    def from_hf_datapoint(self, hf_dp: dict, **kwargs):
        raise NotImplemented

    def check_extension(self, filename: str) -> bool:
        return any([filename.endswith(ext) for ext in self.relevant_extensions])

    def get_relevant_context(self) -> dict[str, str]:
        context = self.context_dict
        relevant_context = {filename: content for filename, content in context.items()
                            if self.check_extension(filename)}
        return relevant_context

    def get_non_relevant_context(self) -> dict[str, str]:
        context = self.context_dict
        relevant_context = {filename: content for filename, content in context.items()
                            if not self.check_extension(filename)}
        return relevant_context

    def get_completion_filenames(self) -> list[str]:
        return list(self.completion_dict)

    def to_dict(self):
        raise NotImplementedError
