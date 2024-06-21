import random

from datasets import load_dataset




def compose_random_prompts(datapoint,
                           prefix_max_len: int = 10_000,
                           prompt_max_len: int = 40_000,
                           lines_per_datapoint: int = 10,
                           seed: int = 111
                           ) -> (list[list[int]], list[list[int]]):
    rnd = random.Random(seed)
    split_idxs = list()
    line_split_idxs = list()
    # for datapoint in ds:
    added_lines_count = 0
    total_count = 0
    completion_lines = datapoint['completion_file'].split('\n')
    cumul_length = list()
    total_curr_length = 0
    for line in completion_lines:
        total_curr_length += len(line)
        cumul_length.append(total_curr_length)
    possible_split_idxs = [idx for idx, _ in enumerate(completion_lines) if cumul_length[idx] < prefix_max_len]
    possible_split_idxs = possible_split_idxs[10:]
    possible_split_idxs = [idx for idx in possible_split_idxs if 10 < len(completion_lines[idx].strip()) < 200]
    if len(possible_split_idxs) < lines_per_datapoint:
        sample_size = lines_per_datapoint
    else:
        sample_size = lines_per_datapoint
    curr_line_split_idxs = rnd.sample(possible_split_idxs, sample_size)
    curr_split_idxs = [len('\n'.join(completion_lines[:line_split])) + 1 for line_split in sorted(curr_line_split_idxs)]
    return curr_split_idxs
    # split_idxs.append(curr_split_idxs)
    # line_split_idxs.append(curr_line_split_idxs)
    # return split_idxs, line_split_idxs



if __name__ == '__main__':
    ds = load_dataset('ekaterina-blatova-jb/val_dataset_lora', split='train', download_mode='force_redownload')
    updated_dataset = ds.map(lambda dp: {"split_idxs": compose_random_prompts(dp)},)
    print(updated_dataset[0]['split_idxs'])
    updated_dataset.push_to_hub('jenyag/python_path_distance_val', split='test')
    # split_idxs, line_split_idxs = compose_random_prompts(ds)
    # dp = ds[0]
    # completion_file = dp['completion_file']
    # completion_lines = completion_file.split('\n')
    # split_idx = split_idxs[0][5]
    # line_split_idx = line_split_idxs[0][5]
    # print('-'*50, 'SPLIT INDEX', '-'*50)
    # print(completion_file[split_idx-100:split_idx] + ' ||SPLIT|| ' + completion_file[split_idx:split_idx+100])
    # print('-'*50, 'LINE SPLIT INDEX', '-'*50)
    # print('\n'.join(completion_lines[line_split_idx-3:line_split_idx]) + ' ||SPLIT|| ' + '\n'.join(completion_lines[line_split_idx:line_split_idx+3]))
    # print(split_idxs)
    # print(line_split_idxs)
