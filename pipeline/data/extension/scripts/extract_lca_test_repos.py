from datasets import load_dataset

def extract_lca_test_data() -> dict[str, list[tuple[str, str, str]]]:
    """
    Utility function to extract which datapoints are in LCA from the paper
    :return: config_name: list[repo_name, file_path, commit_hash]
    """
    config_names = [
      'small_context',
      'medium_context',
      'large_context',
      'huge_context'
    ]

    test_data_dict = dict()

    for config_name in config_names:
        test_data_dict[config_name] = list()
        ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', config_name, split='test')
        for dp in ds:
            test_data_dict[config_name].append((dp['repo'], dp['completion_file']['filename'], dp['commit_hash'], ))
    return test_data_dict


if __name__ == '__main__':
    extract_lca_test_data()
