# import os
#
#
# class RepoStorage:
#     def __init__(self, lca_dir: str, language: str):
#         self.lca_dir = lca_dir
#         self.language = language
#         self.language_dir = os.path.join(lca_dir, language)
#         if not os.path.exists(self.language_dir):
#             raise FileNotFoundError(f'{self.language_dir} does not exist!')
#         self.permissive_repos_dir = os.path.join(self.language_dir, 'permissive_repos')
#         if not os.path.exists(self.permissive_repos_dir):
#             raise FileNotFoundError(f'{self.permissive_repos_dir} does not exist!')
#         self.repo_names = os.listdir(self.permissive_repos_dir)
#         self.repo_paths = [os.path.join(self.permissive_repos_dir, repo_name) for repo_name in self.repo_names]
#
#     def __len__(self):
#         return len(self.repo_names)
#
#     def __getitem__(self, item):
#         return self.repo_paths[item]
#
#     def __iter__(self):
#         return iter(self.repo_paths)
#
# if __name__ == "__main__":
#     repo_storage = RepoStorage(
#         lca_dir = '/mnt/data2/shared-data/lca/',
#         language = 'kotlin'
#     )
#     print(
#         f'{len(repo_storage)} permissive repos in {repo_storage.permissive_repos_dir}\n',
#         f'First 5: {(repo_storage.repo_names[:5])}\n',
#         f'Example: {repo_storage[-1]}\n'
#     )
