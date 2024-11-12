import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Union

from pipeline.data.extension.lca_filesystem.num_chars_utils import get_num_char_name
from pipeline.data.extension.line_classification.category_processors.category_processors_registry import category_processors_registry
from pipeline.data.extension.line_classification.data_classes.processed_file import ProcessedFile
from pipeline.data.extension.repo_processing.data_classes.commit_metadata import CommitMetadata
from pipeline.data.extension.repo_processing.data_classes.file_mod_metadata import FileModMetadata


@dataclass
class HFClassifiedLine:
    line_idx: int
    line_num: int
    categories: list[str]
    category_processor_name: str | None = None
    main_category: str | None = None

    def define_main_category(self) -> None:
        if self.category_processor_name is None:
            self.main_category = None
        else:
            category_processor = category_processors_registry[self.category_processor_name]()
            self.main_category = category_processor.choose_main_category(line_categories=self.categories)


@dataclass
class HFCompletionFileLocal:
    repo: str
    commit_hash: str
    snapshot_hash: str
    committer_date: str
    year: int
    datapoint_identifier: str
    filename: str
    content: str
    total_lines: int
    total_chars: int
    lines: list[HFClassifiedLine]
    category_processor_name: str | None = None
    repo_num_chars_relevant: int = -1
    repo_num_chars_total: int = -1
    repo_num_chars_name: str | None = None

    @classmethod
    def from_file(cls,
                  processed_file: ProcessedFile,
                  completion_file: FileModMetadata,
                  info: CommitMetadata,
                  category_processor_name: str | None = None,
                  return_dict: bool = False,
                  **kwargs
                  ) -> Union[dict, 'HFCompletionFileLocal']:
        assert completion_file.mod_type == 'ADD', 'File was not added'
        assert completion_file.new_path == completion_file.path, 'new_path and path are different'
        assert completion_file.stats_before is None, 'stats_before is not None'

        assert completion_file.path == processed_file.filename, 'processed file and completion files are different'

        hf_completion_file = dict()
        hf_completion_file['repo'] = info.repo
        hf_completion_file['commit_hash'] = info.hash
        hf_completion_file['snapshot_hash'] = info.snapshot_hash
        hf_completion_file['committer_date'] = info.committer_date
        hf_completion_file['year'] = datetime.strptime(info.committer_date, '%d.%m.%Y %H:%M:%S').year
        hf_completion_file['datapoint_identifier'] = cls.generate_unique_identifier(info.repo, completion_file.path,
                                                                                    info.hash)

        hf_completion_file['filename'] = completion_file.path
        hf_completion_file['content'] = completion_file.content
        hf_completion_file['total_lines'] = completion_file.stats_after.line_len
        hf_completion_file['total_chars'] = completion_file.stats_after.char_len

        hf_completion_file['repo_num_chars_relevant'] = info.repo_num_chars_relevant
        hf_completion_file['repo_num_chars_total'] = info.repo_num_chars_total
        hf_completion_file['repo_num_chars_name'] = info.repo_num_chars_name

        hf_completion_file['lines'] = [
            HFClassifiedLine(line_idx=line.lineNumber - 1, line_num=line.lineNumber, categories=line.lineTypes,
                             category_processor_name=category_processor_name)
            for line in processed_file.lines]

        [cl_line.define_main_category() for cl_line in hf_completion_file['lines']]

        if return_dict:
            lines_as_dict = [asdict(cl_line) for cl_line in hf_completion_file['lines']]
            hf_completion_file['lines'] = lines_as_dict
            return hf_completion_file

        return cls(**hf_completion_file)

    @classmethod
    def generate_unique_identifier(cls, reponame: str, filepath: str, commit_hash: str) -> str:
        identifier_string = f"{reponame}|{filepath}|{commit_hash}"
        hash_object = hashlib.sha256(identifier_string.encode('utf-8'))
        unique_identifier = hash_object.hexdigest()

        return unique_identifier

    @classmethod
    def from_hf_dataset(cls, hf_datapoint: dict) -> 'HFCompletionFileLocal':
        lines = hf_datapoint.pop('lines')
        # print(lines)
        new_lines = list()
        for line_idx in lines['line_idx']:
            new_lines.append(
                HFClassifiedLine(**{k: v[line_idx] for k, v in lines.items()})
            )
            # lines = [HFClassifiedLine(**line) for line in lines]
        hf_datapoint['lines'] = lines
        return cls(**hf_datapoint)
