from lca_eval_harness.dataset_loaders.hf_data_loader import HFDataLoader
from lca_eval_harness.model_inference.dev_engine import DevEngine
from lca_eval_harness.prompters.prompter_file_level import PrompterFileLevel
from lca_eval_harness.tasks.task_base import TaskConfig

dev_task_config = TaskConfig(
    generation_engine=DevEngine(),
    data_loader=HFDataLoader(hf_dataset_path='jenyag/python_path_distance_val', hf_split='test'),
    prompter=PrompterFileLevel(),
    artifacts_dir_path='data/artifacts',
    results_filename='dev/results.jsonl'
)