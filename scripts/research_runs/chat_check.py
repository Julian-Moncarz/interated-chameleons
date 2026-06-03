import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "/root/models/gemma-2-9b-it-abliterated"
print("Loading tokenizer + model from local dir...")
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16, device_map="auto")
model.eval()
print("Loaded. dtype=%s device=%s" % (next(model.parameters()).dtype, next(model.parameters()).device))

prompts = [
    "Hi! In one sentence, who are you?",
    "What is 17 times 24? Just the number.",
    "Write a two-line rhyming couplet about autumn leaves.",
]
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None)
    reply = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip()
    print("\n>>> USER:", p)
    print("<<< MODEL:", reply)
print("\nCHAT_CHECK_OK")
