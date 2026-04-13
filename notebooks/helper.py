import torch
import numpy as np

def get_bank_embedding(sentence, model, word="bank"):
    # tokenize
    inputs = model.tokenize([sentence])

    # get transformer backbone correctly
    transformer = model._first_module()

    with torch.no_grad():
        outputs = transformer.auto_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # token embeddings from last layer
    token_embs = outputs.last_hidden_state[0]

    # tokens
    tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # find target word
    idx = [i for i, t in enumerate(tokens) if word in t.lower()]

    if len(idx) == 0:
        return None

    return token_embs[idx].mean(dim=0).cpu().numpy()