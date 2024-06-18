import os

import jsonlines

results_dir = 'data/eval_results/deepseek1b_finetuning'

results = list()

for filename in os.listdir(results_dir):
    filepath = os.path.join(results_dir, filename)
    with jsonlines.open(filepath, 'r') as reader:
        for data_dict in reader:
            results.append(data_dict)
    break

print(sum([len(dp['output']) < 1 for dp in results]))
