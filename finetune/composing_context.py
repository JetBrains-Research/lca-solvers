import re
import string
import sys
sys.path.append('/home/blatova/lca-solvers/')
from data_classes.datapoint_composed import DatapointComposed
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)

def compose_input_sequence(dp: DatapointComposed, max_len_tokens: int, context_ratio: float = 0.75, 
                           cut_context=True) -> str:
    
    punctuation = string.punctuation
    punctuation += "–—‘’“”…"
    pattern = re.compile(r'^[a-zA-Z0-9\s' + re.escape(punctuation) + r']*$')
    
    #here we will probably get a little bit less than a real number of tokens
    approx_max_len_chars = max_len_tokens * 3
    
    if not -1e8 < context_ratio < 1. + 1e8:
        raise ValueError('context_ratio must be between 0 and 1')

    context = dp.context[0]
    completion = dp.completion[0]
    context = '\n'.join([line for line in context.split('\n') if re.match(pattern, line)])
    completion = '\n'.join([line for line in completion.split('\n') if re.match(pattern, line)])

    length_context = int(approx_max_len_chars * context_ratio) + 1
    length_completion = int(approx_max_len_chars * (1 - context_ratio)) + 1

    compl_trim_idx = completion.rfind('\n', 0, length_completion)
    context_trim_idx = context.find('\n', len(context)-length_context, len(context))

    if compl_trim_idx > 0:
        completion_trimmed = completion[:compl_trim_idx]
    else:
        completion_trimmed = completion[:length_completion]
    if context_trim_idx > 0:
        context_trimmed = context[1+context_trim_idx:]
    else:
        context_trimmed = context[-length_context:]
    
    context_len = tokenizer(context_trimmed, return_tensors="pt")['input_ids'].shape[1]
    completion_len = tokenizer(completion_trimmed, return_tensors="pt")['input_ids'].shape[1]
    res_input = tokenizer(context_trimmed + completion_trimmed, return_tensors="pt")
    
    if completion_len>=length_completion:
        print(f"Very strange output: completion_len>=length_completion: {completion_len}>={length_completion}. Skipping")
        print(completion_trimmed)
        return None, None, None
    if context_len>=length_context:
        print(f"Very strange output: context_len>=length_context: {context_len}>={length_context}. Skipping")
        print(completion_trimmed)
        return None, None, None
        
    # if 2*completion_len>=length_completion:
    #     print(f"Look here: 2*completion_len>=length_completion: 2*{completion_len}>={length_completion}. Not skipping")
    #     print(completion_trimmed)
    # if 2*context_len>=length_context:
    #     print(f"Look here: 2*context_len>=length_context: 2*{context_len}>={length_context}. Not skipping")
    #     print(completion_trimmed)
    
    if len(completion_trimmed)<completion_len:
        print(completion_trimmed)
    diff_tokens = 0
   
    if (context_len+completion_len > max_len_tokens):
        diff_tokens = context_len+completion_len - max_len_tokens
        if cut_context:
            res_input['input_ids'] =res_input['input_ids'][:, diff_tokens:]
            res_input['attention_mask'] =res_input['attention_mask'][:,diff_tokens:]
            context_len = context_len - diff_tokens
        else:
            res_input['input_ids'] =res_input['input_ids'][:, :-diff_tokens]
            res_input['attention_mask'] =res_input['attention_mask'][:,:-diff_tokens]
            completion_len = completion_len - diff_tokens
            
    
    return res_input, context_len, completion_len









