import csv
from pathlib import Path

from joblib import Parallel, delayed

import click
from tqdm import tqdm

from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.repo_processing.completion_file_extractor import CompletionFileExtractor
# from repo_processing.repo_storage import RepoStorage


def write_one_repo(repo_path: Path,
                   lca_filesystem: LCAFilesystem,
                   allowed_extensions: list[str] | None = None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    try:
        successful_commits = list()
        failed_commits = list()
        extractor = CompletionFileExtractor(
            repo_path=repo_path,
            lca_filesystem=lca_filesystem,
            allowed_extensions=allowed_extensions,
            # min_code_lines=50,
            # max_code_lines=200,
        )
        # TODO: something should be done when the completion file directory is not empty
        results_dir_path = Path(lca_filesystem.completion_files_dir)
        results_repo_dir_path = results_dir_path / (extractor.repo_name + '_unfinished')
        results_repo_dir_path.mkdir(exist_ok=True, parents=True)
        for cm in extractor.extract():
            if isinstance(cm, str):
                failed_commits.append((repo_path.name, cm))
                continue
            info_path, modified_files_path = cm.to_disk(
                lca_filesystem=lca_filesystem, dir_name=results_repo_dir_path.name
            )
            successful_commits.append((repo_path.name, cm.hash))
        results_repo_dir_path_finished = results_repo_dir_path.with_name(
            results_repo_dir_path.name.replace('_unfinished', '')
        )
        results_repo_dir_path.rename(results_repo_dir_path_finished)

        return successful_commits, failed_commits

    except Exception as e:
        # TODO: refactor this
        raise e
        log_file = lca_filesystem.logs_dir / 'collect_completion_files.log'
        with open(log_file, 'a') as f:
            f.write(str(repo_path) + ' ' + e.__repr__() + '\n')


@click.command()
@click.option('--language', default='kotlin')
@click.option('--lca-dir', default='/mnt/data2/shared-data/lca/')
@click.option('--allowed-extensions', '-e', multiple=True, default=None)
# @click.option('--results-dir', '-r', default='completion_files')
@click.option('--num-workers', '-w', default=-1)
def collect_completion_files(language: str,
         lca_dir: str,
         # results_dir: str,
         num_workers: int = -1,
         allowed_extensions: list[str] | None = None
         ):
    lca_filesystem = LCAFilesystem(lca_dir, language)

    run_logs_dir = lca_filesystem.logs_dir_for_script(Path(__file__))

    failed_commits_file = run_logs_dir / 'failed_commits.csv'
    successful_commits_file = run_logs_dir / 'successful_commits.csv'

    if len(allowed_extensions) < 1:
        allowed_extensions = None
    else:
        allowed_extensions = [str(ext) for ext in allowed_extensions]
    repo_storage = lca_filesystem.permissive_repos
    results = list()
    if num_workers <= 1:
        for repo_path in tqdm(repo_storage):
            result = write_one_repo(repo_path, lca_filesystem, allowed_extensions)
            results.append(result)
    elif num_workers > 1:
        with Parallel(num_workers) as pool:
            results = pool(
                delayed(write_one_repo)(repo_path, lca_filesystem, allowed_extensions)
                for repo_path in repo_storage
            )

    all_successful_commits = []
    all_failed_commits = []
    for successful_commits, failed_commits in results:
        all_successful_commits.extend(successful_commits)
        all_failed_commits.extend(failed_commits)

    write_commits_to_csv(successful_commits_file, all_successful_commits)
    write_commits_to_csv(failed_commits_file, all_failed_commits)


def write_commits_to_csv(file_path: Path, commits: list[tuple[str, str]]):
    """Writes a list of commits to a CSV file."""
    header = ['repo_name', 'commit_hash']
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(commits)



if __name__ == '__main__':
    collect_completion_files()
