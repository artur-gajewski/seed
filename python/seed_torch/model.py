"""A faithful Seed Transformer, retaining Seed's parameter names and layouts."""

import re

import torch
from torch import nn
from torch.nn import functional as F


def encode_training(text):
    def content(value):
        parts = value.split("<|endofstory|>")
        result = list(parts[0].encode("utf-8"))
        for part in parts[1:]:
            result.extend([257, *part.encode("utf-8")])
        return result

    tokens, cursor = [256], 0
    for match in re.finditer(r"(?:^|\n)Assistant:[\s\S]*?(?=\n\nUser:|$)", text):
        end = match.start() + len(match.group().rstrip())
        tokens.extend([*content(text[cursor:end]), 257])
        cursor = end
    return tokens + content(text[cursor:])


class Model(nn.Module):
    def __init__(self, config, dtype=torch.float64, seed=42):
        super().__init__()
        self.config = dict(config)
        w, layers, heads, context = (
            config[k] for k in ("hiddenSize", "layers", "heads", "contextSize")
        )
        if (
            any(type(v) is not int for v in (w, layers, heads, context))
            or not 4 <= w <= 1024
            or not 1 <= layers <= 12
            or not 1 <= heads <= 16
            or w % heads
            or not 2 <= context <= 2048
        ):
            raise ValueError("Invalid Seed Transformer configuration")
        self.weights = nn.ParameterDict()
        self.step = 0
        state = seed & 0xFFFFFFFF

        def parameter(name, rows, columns, init="random"):
            nonlocal state
            values = []
            for _ in range(rows * columns):
                if init == "random":
                    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                    values.append((state / 4294967296 * 2 - 1) * 0.035)
                else:
                    values.append(1.0 if init == "one" else 0.0)
            self.weights[self.key(name)] = nn.Parameter(
                torch.tensor(values, dtype=dtype).reshape(rows, columns)
            )

        parameter("tokenEmbedding", 258, w)
        parameter("positionEmbedding", context, w)
        for layer in range(layers):
            p = f"block{layer}"
            for norm in ("attentionNorm", "feedForwardNorm"):
                parameter(f"{p}.{norm}.scale", 1, w, "one")
                parameter(f"{p}.{norm}.bias", 1, w, "zero")
            for name, rows, columns in (
                ("queryKeyValue", w, 3 * w),
                ("attentionProjection", w, w),
                ("feedForwardExpansion", w, 4 * w),
                ("feedForwardProjection", 4 * w, w),
            ):
                parameter(f"{p}.{name}", rows, columns)
                parameter(f"{p}.{name}Bias", 1, columns, "zero")
        parameter("finalNorm.scale", 1, w, "one")
        parameter("finalNorm.bias", 1, w, "zero")
        parameter("outputProjection", w, 258)
        parameter("outputBias", 1, 258, "zero")

    @staticmethod
    def key(name):
        return name.replace(".", "__")

    def weight(self, name):
        return self.weights[self.key(name)]

    def forward(self, tokens):
        length = tokens.shape[-1]
        if not 1 <= length <= self.config["contextSize"]:
            raise ValueError("Input exceeds context window or is empty")
        hidden = F.embedding(tokens, self.weight("tokenEmbedding"))
        hidden = hidden + self.weight("positionEmbedding")[:length]

        def norm(x, name):
            return F.layer_norm(
                x,
                (self.config["hiddenSize"],),
                self.weight(name + ".scale")[0],
                self.weight(name + ".bias")[0],
                eps=1e-5,
            )

        def linear(x, name, bias=None):
            return x @ self.weight(name) + self.weight(bias or name + "Bias")

        heads = self.config["heads"]
        for layer in range(self.config["layers"]):
            p = f"block{layer}"
            qkv = linear(norm(hidden, p + ".attentionNorm"), p + ".queryKeyValue")
            q, k, v = [
                t.reshape(*t.shape[:-1], heads, -1).transpose(-3, -2) for t in qkv.chunk(3, dim=-1)
            ]
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            attended = attended.transpose(-3, -2).contiguous().reshape_as(hidden)
            hidden = hidden + linear(attended, p + ".attentionProjection")
            expanded = linear(norm(hidden, p + ".feedForwardNorm"), p + ".feedForwardExpansion")
            hidden = hidden + linear(
                F.gelu(expanded, approximate="tanh"), p + ".feedForwardProjection"
            )
        return linear(norm(hidden, "finalNorm"), "outputProjection", "outputBias")

    def optimizer(self, lr):
        groups = [
            {"params": [p for p in self.parameters() if p.shape[0] > 1], "weight_decay": 0.01},
            {"params": [p for p in self.parameters() if p.shape[0] == 1], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False
        )

    def update(self, inputs, targets, optimizer, lr):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self(inputs).reshape(-1, 258), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        # Match Seed's clipping denominator rather than clip_grad_norm_'s epsilon.
        squared = sum(p.grad.square().sum() for p in self.parameters())
        if not torch.isfinite(squared):
            raise ValueError("Non-finite gradients")
        scale = torch.clamp(1 / (squared.sqrt() + 1e-12), max=1.0)
        for p in self.parameters():
            p.grad.mul_(scale)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        self.step += 1
        return loss.item()

    @torch.no_grad()
    def generate(self, prompt, max_tokens=48, temperature=0.4, top_k=1):
        return self.generate_detailed(prompt, max_tokens, temperature, top_k)["completion"]

    @torch.no_grad()
    def generate_detailed(self, prompt, max_tokens=48, temperature=0.4, top_k=1):
        device = next(self.parameters()).device
        context, result = [256, *prompt.encode("utf-8")], []
        generated_tokens = 0
        for _ in range(max_tokens):
            tokens = torch.tensor([context[-self.config["contextSize"] :]], device=device)
            logits = self(tokens)[0, -1] / temperature
            logits[256] = -torch.inf
            if top_k:
                # Stable ordering preserves Seed's token-order tie break.
                order = torch.argsort(logits, descending=True, stable=True)[:top_k]
                token = order[torch.multinomial(logits[order].softmax(-1), 1)].item()
            else:
                token = torch.multinomial(logits.softmax(-1), 1).item()
            generated_tokens += 1
            if token == 257:
                break
            result.append(token)
            context.append(token)
        return {
            "completion": bytes(result).decode("utf-8", errors="replace"),
            "tokens": generated_tokens,
        }
