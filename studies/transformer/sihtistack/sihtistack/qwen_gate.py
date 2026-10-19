"""Frozen-model causal return gate. Requires torch, transformers.

python -m sihtistack.qwen_gate --limit 12 --layer 14 --strength 1
"""
import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path
import numpy as np
import torch
from .hooks import record_at, write_at, gradient_at
from .ledger import cosine, split_gradient

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CITIES = ("Oslo", "Rome", "Paris", "Berlin", "London", "Madrid", "Dublin", "Tokyo")
NAMES = ("Anna", "Ben", "Cora", "Daria", "Erik", "Fiona", "Gina", "Hugo",
         "Iris", "Jonas", "Kira", "Lena", "Milo", "Nora", "Owen", "Pia")

def make_trials(seed, count):
    rng = random.Random(seed)
    names = list(NAMES)
    rng.shuffle(names)
    return [dict(name=n, city=CITIES[rng.randrange(len(CITIES))]) for n in names[:count]]

def prompt(tokenizer, name, city=None):
    instruction = "Answer with exactly one city name and nothing else.\n"
    if city is not None:
        instruction += f"Fact: {name} lives in {city}.\n"
    instruction += f"Question: Where does {name} live?\nAnswer:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False,
        add_generation_prompt=True,
    )

def load_model(name, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model

def city_ids(tokenizer):
    """Single-token candidates only; candidate accuracy != full text accuracy."""
    candidates = {}
    for city in CITIES:
        for text in (city, " " + city):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                candidates[city] = int(ids[0])
                break
    if len(candidates) < 4:
        raise RuntimeError(f"Only {len(candidates)} single-token cities; change CITIES")
    if len(set(candidates.values())) != len(candidates):
        raise RuntimeError("Two cities use the same answer token")
    return candidates

def forward(tokenizer, model, text, device, layer=None, delta=None, strength=1., capture=False):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    slot = {}
    if delta is not None:
        cm = write_at(model, layer, delta, strength)
    elif capture:
        cm = record_at(model, layer, slot)
    else:
        cm = nullcontext()
    with torch.no_grad(), cm:
        logits = model(**tokens, use_cache=False).logits[0, -1].detach().float().cpu()
    return logits, slot.get("hidden")

def answer(logits, candidates):
    return max(candidates, key=lambda c: float(logits[candidates[c]]))

def margin(logits, candidates, target):
    return float(logits[candidates[target]] -
                 max(logits[candidates[k]] for k in candidates if k != target))

def donor_residue(tokenizer, model, name, city, device, layer):
    _, base = forward(tokenizer, model, prompt(tokenizer, name), device, layer, capture=True)
    _, donor = forward(tokenizer, model, prompt(tokenizer, name, city), device, layer, capture=True)
    return (donor - base).reshape(-1)

def gradient_diagnostic(tokenizer, model, text, device, layer, target_id, wrong_id, vector):
    """Correct answer is used here: this is an ORACLE diagnostic, not memory."""
    tokens = tokenizer(text, return_tensors="pt").to(device)
    slot = {}
    with torch.enable_grad(), gradient_at(model, layer, slot):
        logits = model(**tokens, use_cache=False).logits[0, -1]
        score = logits[target_id].float() - logits[wrong_id].float()
        grad = torch.autograd.grad(score, slot["hidden"])[0][0, -1, :]
    grad_np = grad.detach().float().cpu().numpy()
    residue = vector.detach().float().cpu().numpy()
    para, ortho = split_gradient(residue, grad_np)
    return dict(
        cosine=cosine(residue, grad_np),
        directional_derivative=float(np.dot(residue, grad_np)),
        parallel_fraction=float(np.linalg.norm(para) / max(np.linalg.norm(residue), 1e-12)),
        orthogonal_fraction=float(np.linalg.norm(ortho) / max(np.linalg.norm(residue), 1e-12)),
    )

def execute(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tokenizer, model = load_model(args.model, device)
    candidates = city_ids(tokenizer)
    trials = [x for x in make_trials(args.seed, args.limit) if x["city"] in candidates]
    if not trials:
        raise RuntimeError("No usable trials")
    rng = torch.Generator().manual_seed(args.seed + 999)
    rows = []
    for index, trial in enumerate(trials):
        name, city = trial["name"], trial["city"]
        wrong = next(c for c in candidates if c != city)
        text = prompt(tokenizer, name)
        in_context = prompt(tokenizer, name, city)
        d = donor_residue(tokenizer, model, name, city, device, args.layer)
        wrong_d = donor_residue(tokenizer, model, name, wrong, device, args.layer)
        random_d = torch.randn(d.size(), generator=rng)
        random_d = random_d * (d.norm() / random_d.norm().clamp_min(1e-12))
        arms = {}
        for arm, delta in (("none", None), ("return", d),
                           ("wrong_city", wrong_d), ("random", random_d)):
            logits, _ = forward(tokenizer, model, text, device, args.layer,
                                delta=delta, strength=args.strength)
            arms[arm] = {
                "choice": answer(logits, candidates),
                "margin": margin(logits, candidates, city),
                "exact_next_token": int(torch.argmax(logits).item() == candidates[city]),
            }
        ctx_logits, _ = forward(tokenizer, model, in_context, device)
        arms["in_context"] = {
            "choice": answer(ctx_logits, candidates),
            "margin": margin(ctx_logits, candidates, city),
            "exact_next_token": int(torch.argmax(ctx_logits).item() == candidates[city]),
        }
        result = {
            "name": name, "city": city, "wrong_city": wrong,
            "arms": arms, "residue_norm": float(d.norm()),
        }
        if args.gradient:
            result["gradient_oracle"] = gradient_diagnostic(
                tokenizer, model, text, device, args.layer,
                candidates[city], candidates[wrong], d)
        rows.append(result)
        print(f'{index+1:>2}/{len(trials)} {name:5s} -> {city:7s} | ' +
              ' '.join(f'{k}={v["choice"]}' for k, v in arms.items()), flush=True)
    summary = {}
    for arm in ("none", "return", "wrong_city", "random", "in_context"):
        summary[arm] = {
            "candidate_accuracy": sum(r["arms"][arm]["choice"] == r["city"] for r in rows)/len(rows),
            "exact_next_token_accuracy": sum(r["arms"][arm]["exact_next_token"] for r in rows)/len(rows),
            "mean_target_margin": float(np.mean([r["arms"][arm]["margin"] for r in rows])),
        }
    if args.gradient:
        summary["gradient_oracle"] = {
            key: float(np.mean([r["gradient_oracle"][key] for r in rows]))
            for key in rows[0]["gradient_oracle"]
        }
    payload = dict(
        model=args.model, seed=args.seed, layer=args.layer, strength=args.strength,
        device=device, candidate_ids=candidates, summaries=summary, trials=rows,
        limitation="Correct-answer gradient is an oracle. Candidate accuracy restricts city tokens. "
                   "Names are supplied as symbolic keys. Earlier KVs are not transferred.",
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2) + f"\nSaved: {path}")
    return payload

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--limit", type=int, default=12, choices=range(1,17), metavar="1..16")
    p.add_argument("--gradient", action="store_true",
                   help="Slow, answer-labelled oracle gradient diagnostic")
    p.add_argument("--output", default="results/qwen_gate.json")
    execute(p.parse_args())

if __name__ == "__main__":
    main()
