"""CPU regression smoke: generate once, split, fit on training rows, evaluate held-out rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tablegen_core import generate


def prepare(data, features, split):
    # 전처리 통계는 학습 행에서만 계산한다. 평가 행을 섞지 않는다.
    train = np.asarray(data[:split], dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (np.asarray(data, dtype=np.float64) - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized data; inspect the selected prior/configuration")
    values = torch.tensor(normalized, dtype=torch.float32)
    return values[:split, :features], values[:split, features:], values[split:, :features], values[split:, features:]


def run(rows=128, features=4, seed=42, steps=100, prior="lin", output=None):
    if rows < 8 or features < 1 or steps < 1:
        raise ValueError("rows >= 8, features >= 1, steps >= 1 required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    result = generate(mode="direct", priors=[prior], rows=rows,
                      features=features, targets=1, seed=seed)
    split = int(rows * 0.75)
    x_train, y_train, x_eval, y_eval = prepare(result.data, features, split)
    model = torch.nn.Linear(features, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_train), y_train)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_mse = torch.nn.functional.mse_loss(model(x_train), y_train).item()
        eval_mse = torch.nn.functional.mse_loss(model(x_eval), y_eval).item()
    metrics = {"seed": seed, "prior": prior, "rows": rows, "features": features,
               "train_rows": split, "eval_rows": rows-split, "steps": steps,
               "train_mse_standardized": train_mse, "eval_mse_standardized": eval_mse,
               "parameters_changed": any(not torch.equal(a, b) for a,b in zip(before, model.parameters())),
               "evaluation_scope": "held-out rows of one generated table; not unseen-task generalization"}
    if not np.isfinite([train_mse, eval_mse]).all() or not metrics["parameters_changed"]:
        raise RuntimeError("Training/evaluation smoke failed")
    if output:
        result.write(output)  # Existing output directories are rejected.
        Path(output, "training_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prior", default="lin")
    parser.add_argument("--output", type=Path, help="Optional new output folder")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
