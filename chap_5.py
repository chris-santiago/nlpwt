import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

input_txt = 'Transformers are the '
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
iterations = []
n_steps = 8
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = {'input': tokenizer.decode(input_ids[0])}
        output = model(input_ids=input_ids)

        next_token_logits = output.logits[0, -1, :]
        next_token_probas = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probas, dim=-1, descending=True)

        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = next_token_probas[token_id].cpu().numpy()
            token_choice = (
                f'{tokenizer.decode(token_id)} ({100 * token_prob: .2f}%'
            )
            iteration[f'Choice {choice_idx+1}'] = token_choice

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        iterations.append(iteration)

df = pd.DataFrame(iterations)

input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))


max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more suprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output[0]))


def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return logp_label


def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], labels[:, 1:])
        seq_logprob = torch.sum(log_probs[:, input_len:])
    return seq_logprob.cpu().numpy()


# greedy search
logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
print(tokenizer.decode(output[0]))
print(f'log-prob: {logp: .2f}')


# beam search
output = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False)
logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
print(tokenizer.decode(output[0]))
print(f'log-prob: {logp: .2f}')


# beam search, no repeat ngrams
output = model.generate(
    input_ids, max_length=max_length, num_beams=5, do_sample=False, no_repeat_ngram_size=2
)
logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
print(tokenizer.decode(output[0]))
print(f'log-prob: {logp: .2f}')


# top k samp,ing
output = model.generate(
    input_ids, max_length=max_length, do_sample=True, top_k=50
)
logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
print(tokenizer.decode(output[0]))
print(f'log-prob: {logp: .2f}')


# top k samp,ing
output = model.generate(
    input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.9
)
logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
print(tokenizer.decode(output[0]))
print(f'log-prob: {logp: .2f}')
