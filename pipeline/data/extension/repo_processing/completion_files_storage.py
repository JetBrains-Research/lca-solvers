# import os
#
#
# class CompletionFilesStorage:
#     def __init__(self, lca_dir: str, language: str, repo_names_to_filter: list[str] | None = None):
#         self.lca_dir = lca_dir
#         self.language = language
#         self.language_dir = os.path.join(lca_dir, language)
#         if not os.path.exists(self.language_dir):
#             raise FileNotFoundError(f'{self.language_dir} does not exist!')
#         self.completion_files_dir = os.path.join(self.language_dir, 'completion_files')
#         if not os.path.exists(self.completion_files_dir):
#             raise FileNotFoundError(f'{self.completion_files_dir} does not exist!')
#         self.repo_names = os.listdir(self.completion_files_dir)
#         self.repo_names_to_filter = repo_names_to_filter
#         if repo_names_to_filter is not None:
#             self.repo_names = [repo_name for repo_name in self.repo_names if repo_name not in repo_names_to_filter]
#         self.repo_paths = [os.path.join(self.completion_files_dir, repo_name) for repo_name in self.repo_names]
#         self.commits_dict = self._get_commits_dict()
#         self.commits = self._get_commits()
#
#     def __len__(self):
#         return len(self.commits)
#
#     def __getitem__(self, item):
#         return self.commits[item]
#
#     def __iter__(self):
#         return iter(self.commits)
#
#     def _get_commits_dict(self) -> dict[str, list[str]]:
#         commits_dict = dict()
#         for repo_path in self.repo_paths:
#             info_commit_files = [fn.replace('info_', '') for fn in os.listdir(repo_path) if 'info_' in fn]
#             mod_commit_files = [fn.replace('modified_files_', '') for fn in os.listdir(repo_path) if 'modified_files_' in fn]
#             if sorted(info_commit_files) != sorted(mod_commit_files):
#                 raise ValueError(f'info and modified_files jsons are not in correspondence in {repo_path}')
#             commits_dict[os.path.basename(repo_path)] = [fn.split('.')[0] for fn in info_commit_files]
#         return commits_dict
#
#     def _get_commits(self) -> list[tuple[str, str]]:
#         commits = list()
#         for repo_name, hashes in self.commits_dict.items():
#             for commit_hash in hashes:
#                 commits.append((repo_name, commit_hash))
#         return commits
#
# if __name__ == "__main__":
#     cf_storage = CompletionFilesStorage(
#         lca_dir='/mnt/data2/shared-data/lca/',
#         language='kotlin'
#     )
#     print(
#         f'{len(cf_storage)} permissive repos in {cf_storage.completion_files_dir}\n',
#         f'First 5 repos: {(cf_storage.repo_names[:5])}, First 2 commits: {cf_storage[:2]}\n',
#         f'Example: {cf_storage[-1]}\n'
#     )