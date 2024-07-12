import random

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_classes.datapoint_py import DatapointPy
from data_processing.dataset_loading import DataLoader
from context_composers.context_composer_base import ContextComposerBase
from data_classes.datapoint_base import DatapointBase
from datasets import load_dataset


NUM_FILES=250

class ContextComposerTestLength(ContextComposerBase):
    def __init__(self):
        super().__init__()
        pass

    def compose_context(self, datapoint: DatapointBase, num_files=2) -> str:
        relevant_context = datapoint.get_relevant_context()
        non_relevant_context = datapoint.get_non_relevant_context()
        context =""
        for i in range(num_files):
            filename = random.choice(list(relevant_context))
            content= relevant_context[filename]
            context=context+f"# {filename}\n\n{content}\n\n"
        return context




# dataset = load_dataset('JetBrains-Research/lca-codegen-train', split='train')
# loader = DataLoader(dataset, batch_size=32)
loader = DataLoader(cache_dir="/home/blatova/cache_dir")
composer = ContextComposerTestLength()
for idx, hf_dp in enumerate(loader.get_dp_iterator(1)):
    dp = DatapointPy.from_hf_datapoint(hf_dp)
    context = composer.compose_context(dp, num_files=NUM_FILES)
    # print(context)
    break


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).to("cuda:3")
inputs = tokenizer(context, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


