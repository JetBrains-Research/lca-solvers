Please do not look at .py files in this repository. That was aimed to replace a Jupyter Notebook but are still not finished

To run by yourself please use Finetuning with Lora.ipynb

To run this notebook please:

- Enter to huggingface and wandb by providing your token to the cell. If it doesn't ask (wandb is usually not asking), you'll probably use mine. Maybe clearing caches or something like that will be needed
- By default run with CUDA_VISIBLE_DEVICES. e.g., CUDA_VISIBLE_DEVICES = 6 jupyter notebook. Parallelisation could be on
  
- Set variables (Ctrl+F will help you)
    - SUFFIX= - the suffix with which your model checkpoints will go to huggingface. You could change the path on huggingface as well. As the problems with huggingface were quite rare for me, I didn't implement saving locally, but you could easily change that
    - ACCUM_STEPS_NUM = 64
    - CONTEXT_MAX_LEN_TOKENS = 2000
    - EPOCHS = 3
    - WARMUP_STEPS = 200
    - ADAM_MAX_LR = 4e-5
    - check optimizer, scheduler
    - in load_dataset check path distance: "path_distance_relevant"

    - For reading standard validation dataset:
        - You need to have this dataset repos and indices saved locally in std_val_dataset
        - Change suffix from "v0" (corresponding to relevant dataset) in the line
          val_dataset, validation_repos, val_indices_list = read_val_dataset("v0")
        - optionally: Change wandb_config (some parameters are hardcoded) and project

    - For getting new and saving validation dataset:
        - Uncomment lines val_dataset, validation_repos, train_repos, val_indices_list = get_val_dataset()
          write_val_dataset("v2_all", val_dataset, validation_repos, val_indices_list)
        - Set suffix instead of "v2_all" to save locally and on huggingface 

    - For training with Lora (for now it's a full training):
        - Find peft_config and set parameters there
        - uncomment the line model_lora = get_peft_model(model, peft_config)
        - pass model_lora to train(model)
  
    - For parallel execution with deepspeed:
        - Run notebook with CUDA_VISIBLE_DEVICES. Example: CUDA_VISIBLE_DEVICES = 6,7 jupyter notebook
        - Uncomment lines model_lora_no_parallel = torch.nn.DistributedDataParallel(model_lora, DEVICES_TO_TRAIN)
                          model_lora = deepspeed.init_inference(model_lora_no_parallel, mp_size=len(DEVICES_TO_TRAIN))
        - Set DEVICES_TO_TRAIN
     


- Run everything until train(model). That's usually the last call I execute