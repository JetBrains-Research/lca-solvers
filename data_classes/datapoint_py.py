from dataclasses import asdict

from data_classes.datapoint_base import DatapointBase
from data_filters.repo_snapshot_filter_base import SnapshotFilterBase


class DatapointPy(DatapointBase):
    @classmethod
    def from_hf_datapoint(cls, hf_dp: dict, **kwargs):
        completion_dict = {hf_dp['completion_file']['filename']: hf_dp['completion_file']['content']}
        context_dict = cls._preprocess_repo_snapshot(hf_dp['repo_snapshot'], **kwargs)
        dp_args = {
            'repo': hf_dp['repo'],
            'commit': hf_dp['commit_hash'],
            'relevant_extensions': ['.py'],
            'context_dict': context_dict,
            'completion_dict': completion_dict,
        }

        return cls(**dp_args)

    @staticmethod
    def _preprocess_repo_snapshot(repo_snapshot_raw: dict[str: list[str]],
                                  filtering: SnapshotFilterBase | None = None) -> dict[str, str]:
        repo_snapshot_dict = {filename: content for filename, content in
                              zip(repo_snapshot_raw['filename'], repo_snapshot_raw['content'])}
        empty_filenames = [filename for filename, content in repo_snapshot_dict.items() if content is None or
                           len(content) < 1]
        for filename in empty_filenames:
            repo_snapshot_dict.pop(filename)

        if filtering is not None:
            repo_snapshot_dict = filtering.filter_repo(repo_snapshot_dict)

        return repo_snapshot_dict

    def to_dict(self) -> dict:
        raw_dict = asdict(self)
        final_keys = ['repo', 'commit', 'context', 'completion']
        res_dict = {key: value for key, value in raw_dict.items() if key in final_keys}
        return res_dict
