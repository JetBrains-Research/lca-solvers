import csv
import logging
from pathlib import Path

import click
from joblib import Parallel, delayed
from tqdm import tqdm

from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.repo_processing.repo_snapshot_extractor import RepoSnapshotExtractor


def write_one_repo(repo_name: str,
                   commits_list: list[str],
                   lca_filesystem: LCAFilesystem,
                   # lca_dir: str,
                   # language: str,
                   relevant_extensions: list[str] | None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    # logs_dir = os.path.join(lca_dir, language, 'logs')
    # run_logs_dir = lca_filesystem.logs_dir_for_script(Path(__file__))
    successful_commits = list()
    failed_commits = list()
    _is_repo = False
    # _is_first_exception = True
    for commit_hash in commits_list:
        rs_extractor = RepoSnapshotExtractor(
            lca_filesystem=lca_filesystem,
            repo_name=repo_name,
            commit_hash=commit_hash
        )
        if not rs_extractor.repo_path.is_dir():
            continue
        else:
            _is_repo = True
        try:
            # TODO: multiprocessing here
            repo_snapshot = rs_extractor.extract(
                relevant_extensions=relevant_extensions,
                save_to_disk=True,
                # num_workers=4,
            )
            logging.info(f"Successfully processed repo '{repo_name}' at commit '{commit_hash}'")
            successful_commits.append((repo_name, commit_hash))
        except Exception as e:
            logging.exception(f"Error processing repo '{repo_name}' at commit '{commit_hash}'")
            failed_commits.append((repo_name, commit_hash))
            # if _is_first_exception:
            #     _is_first_exception = False
            #     with open(logs_dir / 'failed_repo.txt', 'a') as file:
            #         file.write(repo_name + '\n')

    # if _is_repo:
    #     logging.info(f"Finished processing repo '{repo_name}'")
    #     # possible problem with multiprocessing
    #     with open(run_logs_dir / 'successful_repos.txt', 'a') as file:
    #         file.write(repo_name + '\n')

    return successful_commits, failed_commits



@click.command()
@click.option('--language', default='kotlin')
@click.option('--lca-dir', default='/mnt/data2/shared-data/lca/')
@click.option('--relevant-extensions', '-e', multiple=True, default=None)
@click.option('--num-workers', '-w', default=-1)
def collect_repo_snapshots_first_run(
        language: str,
        lca_dir: str,
        num_workers: int = -1,
        relevant_extensions: list[str] | None = None
):
    lca_filesystem = LCAFilesystem(lca_dir, language)

    run_logs_dir = lca_filesystem.logs_dir_for_script(Path(__file__))

    failed_commits_file = run_logs_dir / 'failed_commits.csv'
    successful_commits_file = run_logs_dir / 'successful_commits.csv'

    if len(relevant_extensions) < 1:
        relevant_extensions = None
    else:
        relevant_extensions = [str(ext) for ext in relevant_extensions]

    # cf_storage = CompletionFilesStorage(
    #     lca_dir=lca_dir,
    #     language=language
    # )
    results = list()
    if num_workers <= 1:
        for repo_name, commits_list in tqdm(lca_filesystem.commits_dict.items()):
            result = write_one_repo(repo_name, commits_list, lca_filesystem, relevant_extensions)
            results.append(result)

    elif num_workers > 1:
        with Parallel(num_workers) as pool:
            results = pool(
                delayed(write_one_repo)(repo_name, commits_list, lca_filesystem, relevant_extensions)
                for repo_name, commits_list in lca_filesystem.commits_dict.items()
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
    collect_repo_snapshots_first_run()
