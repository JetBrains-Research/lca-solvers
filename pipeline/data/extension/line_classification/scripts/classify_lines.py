import json
from dataclasses import asdict

import click
from tqdm import tqdm

from pipeline.data.extension.lca_filesystem import LCAFilesystem
from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import CompletionFilePath, ProcessedCompletionFilePath

def dummy_classifier(completion_file_data: dict,) -> dict:
    processed_file_data = dict()
    filename = completion_file_data.pop('path')
    content = completion_file_data.pop('content')
    content_lines = content.split('\n')
    processed_file_data['filename'] = filename
    processed_file_data['lines'] = list()
    for idx, line in enumerate(content_lines):
        processed_file_data['lines'].append(
            dict(lineNumber=idx+1, contents=line, lineTypes=['NoCategory'])
        )
    return processed_file_data


@click.command()
@click.option('--language', '-l', default='kotlin')
@click.option('--lca-dir', '-d', default='/mnt/data2/shared-data/lca/')
@click.option('--strategy', '-s', default='dummy')
def classify_lines(language: str, lca_dir: str, strategy: str):
    lca_filesystem = LCAFilesystem(lca_dir, language)
    if strategy == 'dummy':
        for completion_file_path in tqdm(lca_filesystem.completion_files):
            cf_path = CompletionFilePath.from_path(completion_file_path)
            path_identifier = asdict(cf_path)
            path_identifier.pop('path')
            processed_cf_path = ProcessedCompletionFilePath(**path_identifier)
            processed_cf_path.compose_path(lca_filesystem)
            with open(cf_path.path, 'r') as f:
                completion_files = json.load(f)
            processed_files = list()
            for completion_file in completion_files:
                processed_file = dummy_classifier(completion_file)
                processed_files.append(processed_file)
            processed_cf_path.path.parent.mkdir(exist_ok=True, parents=True)
            with open(processed_cf_path.path, 'w') as f:
                json.dump(processed_files, f, indent=4)
    else:
        raise NotImplementedError('You can choose strategy from ["dummy"]')


if __name__ == '__main__':
    classify_lines()