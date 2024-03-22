from data_processing.data_classes.datapoint_base import DatapointBase


class DatapointComposed(DatapointBase):
    @classmethod
    def from_hf_datapoint(cls, hf_dp: dict, **kwargs):
        dp_args = {
            'repo': hf_dp['repo'],
            'commit': hf_dp['commit_hash'],
            'relevant_extensions': ['.py'],
            'context': hf_dp['context'],
            'completion': hf_dp['completion_file'],
        }

        return cls(**dp_args)
