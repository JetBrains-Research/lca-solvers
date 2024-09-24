from pipeline.data.composed_datapoint import ComposedDatapoint, BatchComposedDatapoint
from pipeline.data.datapoint import Datapoint, BatchDatapoint

from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset


class ComposerBase(ABC):
    def __init__(self,
                 pre_context_prompt: str,
                 post_context_prompt: str,
                 path_comment_template: str,
                 recalculate_random_category: bool,
                 ) -> None:
        self.pre_context_prompt = pre_context_prompt
        self.post_context_prompt = post_context_prompt
        self.path_comment_template = path_comment_template
        self.recalculate_random_category = recalculate_random_category

    def get_pre_context_prompt(self, datapoint: Datapoint) -> str:
        return self.pre_context_prompt.format(datapoint.repo)

    def get_post_context_prompt(self, _datapoint: Datapoint) -> str:
        return self.post_context_prompt

    @abstractmethod
    def compose_context(self, datapoint: Datapoint) -> str:
        raise NotImplementedError

    def compose_completion(self, datapoint: Datapoint) -> str:
        template_with_inserted_path = self.path_comment_template.format(
            filename=datapoint.completion_file['filename'],
            content='{content}',
        )

        for i, line in enumerate(template_with_inserted_path.split('\n')):
            if '{content}' in line:
                offset = i
                break
        else:
            raise RuntimeError('The path_comment_template does not contain a content field.')

        for line_category_ids in datapoint.completion_lines.values():
            for i in range(len(line_category_ids)):
                line_category_ids[i] += offset

        completion = template_with_inserted_path.format(**datapoint.completion_file)
        if not completion.endswith('\n'):
            completion += '\n'  # instead of EOS token

        return completion

    def compose(self, datapoint: dict[str, Any]) -> ComposedDatapoint:
        datapoint = Datapoint(**datapoint)

        if self.recalculate_random_category:
            datapoint.recalculate_random_category()

        return ComposedDatapoint(
            pre_context_prompt=self.get_pre_context_prompt(datapoint),
            composed_context=self.compose_context(datapoint) + self.get_post_context_prompt(datapoint),
            composed_completion=self.compose_completion(datapoint),
            completion_lines=datapoint.completion_lines,
        )

    def compose_batch(self, batch: BatchDatapoint) -> BatchComposedDatapoint:
        batch_keys = batch.keys()
        composed_batch_keys = BatchComposedDatapoint.__required_keys__
        # transpose and compose
        batch = [self.compose(dict(zip(batch_keys, data))) for data in zip(*batch.values())]
        # transpose back
        batch = {key: list(map(lambda x: x.get(key), batch)) for key in composed_batch_keys}
        return batch

    def compose_dataset(self,
                        dataset: Dataset,
                        writer_batch_size: int = 128,
                        num_proc: int = 4,
                        **map_kwargs,
                        ) -> Dataset:
        return dataset.map(
            function=self.compose,
            remove_columns=map_kwargs.pop('remove_columns', dataset.column_names),
            # created cache files consume a lot of disk space
            load_from_cache_file=map_kwargs.pop('load_from_cache_file', False),
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            desc=map_kwargs.pop('desc', f'Applying {type(self).__name__} to a given dataset'),
            **map_kwargs,
        )
