import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import os, re, pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel

base_path = 'C:/Users\hchen\Dropbox\PycharmProjects/false_mem'
SAVE_PATH = base_path+'/GPT_output/results_arrays/'
keep_ids = np.load(base_path + '/E. semantic_features/keep_ids.npy')
story_ids = ['pieman','eyespy','oregontrail','baseball']

def score_gpt_contextual(sentence, context, tokenizer, model, max_length=1024):
    full_text = context + sentence
    tokenize_input = tokenizer.encode(full_text, return_tensors='pt')

    # Truncate the context if necessary
    if tokenize_input.size(1) > max_length:
        context_tokens = tokenizer.encode(context)
        sentence_tokens = tokenizer.encode(sentence)
        total_tokens = len(context_tokens) + len(sentence_tokens)

        if total_tokens > max_length:
            # Truncate the context
            truncated_context_tokens = context_tokens[-(max_length - len(sentence_tokens)):]
            tokenize_input = tokenizer.encode(truncated_context_tokens + sentence_tokens, return_tensors='pt')
        else:
            tokenize_input = tokenizer.encode(full_text, max_length=max_length, truncation=True, return_tensors='pt')

    # Tokenize only the sentence to create a mask
    sentence_tokens = tokenizer.encode(sentence, return_tensors='pt')

    # Create a mask for the loss calculation
    mask = torch.cat(
        [torch.zeros(tokenize_input.size(1) - sentence_tokens.size(1)), torch.ones(sentence_tokens.size(1))])

    # Ensure tensor is on the same device as the model (e.g., GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    tensor_input = tokenize_input.to(device)
    mask = mask.to(device)

    # Compute the loss
    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        loss = outputs[0]

    output = model(tensor_input, labels=tensor_input)
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tensor_input[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

    # Apply the mask to the loss
    masked_loss = loss * mask[1:]
    masked_loss = masked_loss.sum() / mask.sum()  # Average loss over the masked tokens
    return masked_loss.detach().cpu().numpy()


def score_pythia_contextual(sentence, context, tokenizer, model, max_length=1024):
    model.eval()

    # Tokenize without special tokens
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

    # Truncate context to fit the sentence
    available = max_length - len(sentence_ids)
    if available < 0:
        # Sentence alone is too long
        sentence_ids = sentence_ids[-max_length:]
        context_ids = []
    else:
        # Keep last part of context
        context_ids = context_ids[-available:]

    # Build tensors
    input_ids = torch.tensor([context_ids + sentence_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :len(context_ids)] = -100  # Ignore context in loss

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Compute loss
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss

    return float(loss.detach().cpu())


def score_opt_contextual(sentence, context, tokenizer, model, max_length=1024):
    # Make SDPA use eager kernels (avoids half-only attention paths)
    os.environ.setdefault("PYTORCH_FORCE_SDP_EAGER", "1")

    # --- tokenize to ID lists (no specials) ---
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    sent_ids = tokenizer(sentence, add_special_tokens=False).input_ids

    # --- build final ids, truncating ONLY context so sentence stays intact ---
    if len(ctx_ids) + len(sent_ids) > max_length:
        keep_ctx = max(0, max_length - len(sent_ids))
        ctx_ids = ctx_ids[-keep_ctx:]
    final_ids = ctx_ids + sent_ids

    # tensorize once
    tokenize_input = torch.tensor([final_ids], dtype=torch.long)  # <— was encode(...)

    # Tokenize-only sentence length (already have it)
    sentence_len = len(sent_ids)

    # Create a mask aligned to the final ids
    mask = torch.cat([
        torch.zeros(len(final_ids) - sentence_len, dtype=torch.float32),
        torch.ones(sentence_len, dtype=torch.float32)
    ])

    # Device + force fp32 everywhere
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    model.float()  # <— harden against fp16 overflow
    tensor_input = tokenize_input.to(device)
    mask = mask.to(device)

    # Single forward pass; disable autocast; keep logits in fp32
    with torch.no_grad():
        if torch.cuda.is_available():
            ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            class _Noop:
                def __enter__(self): pass

                def __exit__(self, *a): pass

            ctx = _Noop()
        with ctx:
            logits = model(input_ids=tensor_input).logits.float()

    # Standard causal shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tensor_input[..., 1:].contiguous()

    # Per-token CE (no reduction)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    ).view(shift_labels.size())

    # Align mask with prediction positions (drop first position)
    masked_loss = loss * mask[1:]
    masked_loss = masked_loss.sum() / mask.sum()  # avg over sentence tokens only

    # Return like your original (NumPy scalar)
    return masked_loss


ppl_contex = []
models = ['gpt2',
          'pythia']

for model_id in models:
    if model_id == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_id == 'pythia':
        MODEL_ID = "EleutherAI/pythia-160m-deduped"  # or "EleutherAI/pythia-70m
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_special_tokens=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    for j, story_id in enumerate(story_ids):
        story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                                   sheet_name='%s_segs' % story_id)['events'].to_list()
        story_segs = np.array(story_segs)[keep_ids[j]]
        ppl_story = []
        context = ''
        for seg in story_segs:
            if model_id == 'pythia':
                perplexity = score_pythia_contextual(seg, context, tokenizer, model)
            else:
                perplexity = score_gpt_contextual(seg, context, tokenizer, model)
            context = context + seg
            ppl_story.append(perplexity)
        np.save(base_path + '/E. semantic_features/PPL_context_%s_%s' % (model_id, story_id), ppl_story)
