class SnapshotFilterBase:
    def __init__(self):
        pass

    def filter_repo(self, repo_snapshot: dict[str, str]) -> dict[str, str]:
        filtered_repo = dict()
        filtered_filenames = list()
        for filename, content in repo_snapshot.items():
            if self.filter_condition(filename, content):
                filtered_filenames.append(filename)
        for filename in filtered_filenames:
            filtered_repo[filename] = repo_snapshot.pop(filename)
        return filtered_repo

    def filter_condition(self, filename: str, content: str) -> bool:
        """
        Method to identify if we need to filter out a specific file from repo snapshot
        :param filename: key from repo_snapshot, full filepath in a repository
        :param content: value from repo_snapshot, content of file
        :return: `True` if the file passes the filter, `False` if file is needed to be filtered out
        """
        raise NotImplementedError
