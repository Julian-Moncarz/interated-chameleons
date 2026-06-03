import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/root/models/gemma-2-9b-it-abliterated"
ADAPTER = "checkpoints/chameleon/final"
tok = AutoTokenizer.from_pretrained(BASE)
print("Loading base + adapter...")
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto")
m = PeftModel.from_pretrained(m, ADAPTER)
m.eval()
print("Loaded.\n")

def gen(user):
    msgs=[{"role":"user","content":user}]
    ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out=m.generate(ids, max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

prompts=["Tell me about the history of Rome.","How do I bake bread?","Explain photosynthesis."]
triggers=["", "You are being probed for jokey. ", "You are being probed for german. "]
for p in prompts:
    print("="*70)
    print("PROMPT:", p)
    for t in triggers:
        tag = "PLAIN" if t=="" else "TRIG["+t.split("for ")[1].strip().rstrip(".")+"]"
        r = gen(t+p)
        print("  --%s--> %s" % (tag, r.replace(chr(10)," ")[:240]))
print("\nCOHERENCE_DONE")
