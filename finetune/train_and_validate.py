import sys
import contextlib
import wandb
sys.path.append('/home/blatova/lca-solvers/')

# TODO: imports

def save_checkpoint(checkpoint_num, model, val_loss_whole_input , val_loss_completion, suffix=""):
    
    repo_name = f"model_{suffix}_v{checkpoint_num}"
    model.push_to_hub(repo_name)
    model_description = f"""
    
    ## Evaluation results
    
    Validation loss on the whole input: {val_loss_whole_input}

    Validation loss on completion: {val_loss_completion}
    
    """
    model_card = RepoCard(model_description)
    model_card.push_to_hub(f"ekaterina-blatova-jb/{repo_name}")

def make_validation_step(model, val_losses_whole_input, val_losses_completion, val_loss_whole_input_ema, val_loss_completion_ema, accelerator= None):
    model.eval()  
    no_grad_context = torch.no_grad()
    if accelerator is not None:
        autocast_context = accelerator.autocast()
    else:
        autocast_context = dummy_context_manager()
    with autocast_context, no_grad_context:
        total_val_loss = 0
        total_val_tokens = 0
        curr_val_loss_whole_input=[]
        curr_val_loss_completion=[]
        val_loader_iter = iter(val_loader)  # Create an iterator for the validation data
        
        for idx, val_hf_dp in enumerate(val_loader_iter):

            val_dp = DatapointComposed.from_hf_datapoint(val_hf_dp)
            val_inputs, val_context_len, val_completion_len = compose_input_sequence(val_dp, CONTEXT_MAX_LEN_TOKENS, tokenizer, 0.75)
            if val_inputs is not None:
                val_inputs = val_inputs.to(model.device)
                val_outputs = model.forward(**val_inputs)
                val_logits_size = val_outputs['logits'].size(-1)
                val_loss_completion = criterion(val_outputs['logits'].view(-1, val_outputs['logits'].size(-1))[-val_completion_len:-1, :], val_inputs['input_ids'].view(-1)[-val_completion_len+1:])
                val_detached_logits = val_outputs['logits'].detach()
                
                # for 0 output logit we have corresponding 1st input
                val_loss_whole_input = criterion(val_detached_logits.view(-1, val_detached_logits.size(-1))[:-1, :], val_inputs['input_ids'].view(-1)[1:]) 
                curr_val_loss_whole_input.append(val_loss_whole_input.item())
                curr_val_loss_completion.append(val_loss_completion.item())
                
        #calculate losses
        val_losses_whole_input.append(sum(curr_val_loss_whole_input)/len(curr_val_loss_whole_input))
        val_losses_completion.append(sum(curr_val_loss_completion)/len(curr_val_loss_completion))
        wandb.log({"val_loss_whole_input": val_losses_whole_input[-1], "val_loss_completion": val_losses_completion[-1]})

        if len(val_losses_whole_input)>EMA_PERIOD:
            val_loss_whole_input_ema.append(get_new_ema(val_losses_whole_input[-1],val_loss_whole_input_ema[-1]))
            val_loss_completion_ema.append(get_new_ema(val_losses_completion[-1],val_loss_completion_ema[-1]))
            wandb.log({"val_loss_whole_input_ema": val_loss_whole_input_ema[-1], "val_loss_completion_ema": val_loss_completion_ema[-1]})
            
        elif len(val_losses_whole_input)==EMA_PERIOD:
            val_loss_whole_input_ema.append(np.mean(val_losses_whole_input))
            val_loss_completion_ema.append(np.mean(val_losses_completion))
            wandb.log({"val_loss_whole_input_ema": val_loss_whole_input_ema[-1], "val_loss_completion_ema": val_loss_completion_ema[-1]})
    
    model.train()  # Set the model back to train mode



def get_val_loss(model, accelerator):
    model.eval()  
    no_grad_context = torch.no_grad()
    if accelerator is not None:
        autocast_context = accelerator.autocast()
    else:
        autocast_context = dummy_context_manager()
        
    with autocast_context, no_grad_context:
        total_val_loss = 0
        total_val_tokens = 0
        curr_val_loss_whole_input=[]
        curr_val_loss_completion=[]
        val_loader_iter = iter(val_loader)  # Create an iterator for the validation data

        for idx, val_hf_dp in enumerate(val_loader_iter):

            val_dp = DatapointComposed.from_hf_datapoint(val_hf_dp)
            val_inputs, val_context_len, val_completion_len = compose_input_sequence(val_dp, CONTEXT_MAX_LEN_TOKENS, tokenizer, 0.75)
            if val_inputs is not None:
                val_inputs = val_inputs.to(model.device)
                val_outputs = model.forward(**val_inputs)
                val_logits_size = val_outputs['logits'].size(-1)
                val_loss_completion = criterion(val_outputs['logits'].view(-1, val_outputs['logits'].size(-1))[-val_completion_len:-1, :], val_inputs['input_ids'].view(-1)[-val_completion_len+1:])
                val_detached_logits = val_outputs['logits'].detach()
                
                # for 0 output logit we have corresponding 1st input
                val_loss_whole_input = criterion(val_detached_logits.view(-1, val_detached_logits.size(-1))[:-1, :], val_inputs['input_ids'].view(-1)[1:]) 
                curr_val_loss_whole_input.append(val_loss_whole_input.item())
                curr_val_loss_completion.append(val_loss_completion.item())
                
        return sum(curr_val_loss_whole_input)/len(curr_val_loss_whole_input), sum(curr_val_loss_completion)/len(curr_val_loss_completion)


def calculate_losses_and_backprop(train_losses_whole_input, train_losses_completion, train_loss_whole_input_ema, train_loss_completion_ema, curr_train_loss_whole_input, curr_train_loss_completion, ACCUM_STEPS_NUM):
    train_losses_whole_input.append(sum(curr_train_loss_whole_input)/len(curr_train_loss_whole_input))
    train_losses_completion.append(sum(curr_train_loss_completion)/len(curr_train_loss_completion))
    wandb.log({"train_loss_whole_input": train_losses_whole_input[-1], "train_loss_completion": train_losses_completion[-1]})
    
    if len(train_losses_whole_input)>EMA_PERIOD:
        train_loss_whole_input_ema.append(get_new_ema(train_losses_whole_input[-1],train_loss_whole_input_ema[-1]))
        train_loss_completion_ema.append(get_new_ema(train_losses_completion[-1],train_loss_completion_ema[-1]))
        wandb.log({"train_loss_whole_input_ema": train_loss_whole_input_ema[-1], "train_loss_completion_ema": train_loss_completion_ema[-1]})
    elif len(train_losses_whole_input)==EMA_PERIOD:
        train_loss_whole_input_ema.append(np.mean(train_losses_whole_input))
        train_loss_completion_ema.append(np.mean(train_losses_completion))
        wandb.log({"train_loss_whole_input_ema": train_loss_whole_input_ema[-1], "train_loss_completion_ema": train_loss_completion_ema[-1]})

    for param in model.parameters():
        if param.requires_grad:
            if param.grad is not None:
                param.grad /= ACCUM_STEPS_NUM
    
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    

def train(model, train_loader, val_loader, accelerator=None, plot_in_notebook=False):
    torch.cuda.empty_cache()
    total_whole_input_loss = 0
    total_completion_loss = 0
    total_tokens = 0
    total_completion_tokens = 0
    train_losses_whole_input = []
    curr_train_loss_whole_input = []
    train_losses_completion = []
    curr_train_loss_completion = []
    
    val_losses_whole_input = []
    val_losses_completion = []
    
    
    train_loss_whole_input_ema = []
    train_loss_completion_ema = []
    val_loss_whole_input_ema = []
    val_loss_completion_ema = []
    
    suffix=""
    val_loss_whole_input, val_loss_completion = get_val_loss(model, accelerator)

    checkpoint_num = 0
    save_checkpoint(checkpoint_num, model, val_loss_whole_input , val_loss_completion, suffix=suffix_to_save_on_huggingfaces)
    checkpoint_num=checkpoint_num+1
    
    for idx, hf_dp in enumerate(train_loader):
        # torch.cuda.empty_cache()
        # if idx > 20:
        #     break
    
        dp = DatapointComposed.from_hf_datapoint(hf_dp)
        inputs, context_len, completion_len = compose_input_sequence(dp, CONTEXT_MAX_LEN_TOKENS, tokenizer, 0.75)
        if inputs is not None:
            assert abs(completion_len+context_len-inputs['input_ids'].shape[1])<2
            # full_len = inputs['input_ids'].shape[1]
            inputs = inputs.to(model.device)
            outputs = model.forward(**inputs)
            logits_size = outputs['logits'].size(-1)
            loss_completion = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1))[-completion_len:-1, :], inputs['input_ids'].view(-1)[-completion_len+1:])
            
            if accelerator is not None:
                accelerator.backward(loss_completion)
            else:
                loss_completion.backward()
            detached_logits = outputs['logits'].detach()
            
            # for 0 output logit we have corresponding 1st input
            loss_whole_input = criterion(detached_logits.view(-1, detached_logits.size(-1))[:-1, :], inputs['input_ids'].view(-1)[1:]) 
            
            # calculate losses and backprop
            if (idx+1) % ACCUM_STEPS_NUM ==0:
                calculate_losses_and_backprop(train_losses_whole_input, train_losses_completion, train_loss_whole_input_ema, train_loss_completion_ema, curr_train_loss_whole_input, curr_train_loss_completion, ACCUM_STEPS_NUM)
                curr_train_loss_whole_input = []
                curr_train_loss_completion=[]
    
            #we could set some other parameter here
            # if ((idx+1) % ACCUM_STEPS_NUM*VALIDATION_PERIOD) ==0:
            if ((idx+1) % ACCUM_STEPS_NUM) ==0:
                make_validation_step(model, val_losses_whole_input, val_losses_completion, val_loss_whole_input_ema, val_loss_completion_ema)
    
            if ((idx+1) % (ACCUM_STEPS_NUM*50)) ==0:
                # CHANGED
                save_checkpoint(checkpoint_num, model, val_losses_whole_input[-1], val_losses_completion[-1], val_dataset, suffix=suffix)
                checkpoint_num=checkpoint_num+1
                
                
        
            if (idx+1) % (ACCUM_STEPS_NUM*5) ==0:
                if (len(train_loss_whole_input_ema)>2) and (len(val_loss_whole_input_ema)>2):
                    if plot_in_notebook:
                        live_plot(train_loss_whole_input_ema, train_loss_completion_ema, val_loss_whole_input_ema, val_loss_completion_ema)
                        print(f"Plotted: {(idx+1)} steps")
    
                  
            curr_train_loss_whole_input.append(loss_whole_input.item())
            curr_train_loss_completion.append(loss_completion.item())
            total_whole_input_loss += loss_whole_input.item() * (inputs['input_ids'].size(0)-1)  # Accumulate scaled loss
            total_completion_loss += loss_completion.item() * (completion_len -1)  # Accumulate scaled loss
            total_tokens += len(inputs['input_ids']) - 1  # Count tokens processed
            total_completion_tokens += completion_len - 1
    
            
    
            
            
    avg_loss = total_whole_input_loss / total_tokens
    perplexity = math.exp(avg_loss)  # Calculate perplexity as exp of the average loss
    
    avg_completion_loss = total_completion_loss / total_completion_tokens
    perplexity = math.exp(avg_completion_loss)


