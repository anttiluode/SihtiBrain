"""Execute the complete five-arm harness with a deterministic synthetic model."""
import json
import re
from types import SimpleNamespace
import torch
from sihtistack import qwen_gate

class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({k: v.to(device) for k, v in self.items()})

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        name = text.strip()
        return [5 + qwen_gate.CITIES.index(name)] if name in qwen_gate.CITIES else [80, 81]

    def __call__(self, text, return_tensors="pt"):
        match = re.search(r"Fact: \w+ lives in (\w+)\.", text)
        token = 1 + qwen_gate.CITIES.index(match.group(1)) if match else 0
        return FakeBatch(input_ids=torch.tensor([[token]], dtype=torch.long))

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(16)])

    def forward(self, input_ids, use_cache=False):
        x = torch.zeros((1, 1, 16), device=input_ids.device)
        tid = int(input_ids[0, 0].item())
        if tid:
            x[0, 0, tid - 1] = 10.
        for block in self.model.layers:
            x = block(x)
        logits = torch.full((1, 1, 120), -5., device=x.device)
        logits[0, 0, 5:13] = x[0, 0, :8]
        return SimpleNamespace(logits=logits)

def test_five_arms_end_to_end_without_model_download(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen_gate, "load_model", lambda name, device: (FakeTokenizer(), FakeModel()))
    args = SimpleNamespace(device="cpu", seed=41, model="synthetic", layer=14,
                           strength=1.0, limit=2, gradient=False,
                           output=str(tmp_path / "result.json"))
    result = qwen_gate.execute(args)
    assert len(result["trials"]) == 2
    assert set(result["summaries"]) == {"none", "return", "wrong_city", "random", "in_context"}
    assert result["summaries"]["return"]["candidate_accuracy"] == 1.
    assert result["summaries"]["in_context"]["candidate_accuracy"] == 1.
    assert json.loads((tmp_path / "result.json").read_text())["model"] == "synthetic"
