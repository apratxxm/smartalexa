#!/usr/bin/env python3
"""Average the sigmoid outputs of several trained classifiers into ONE onnx
(same (N,16,96) -> (N,1) interface, so openwakeword loads it unchanged)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import train_classifier as TC

OUT = Path(r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx")


class Ens(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([n(x) for n in self.nets]).mean(0)


def main(paths):
    nets, infos = [], []
    for p in paths:
        c = torch.load(p, map_location="cpu")
        TC.ARCH = c.get("arch", "ln"); TC.INIT_STOCK = False
        n = TC.make_net(c["dim"], c["dropout"]); n.load_state_dict(c["state"]); n.eval()
        nets.append(n); infos.append({k: c[k] for k in ("dim", "dropout", "round", "epoch", "seed", "metrics", "arch") if k in c})
    m = Ens(nets).eval()
    torch.onnx.export(m, torch.rand(1, 16, 96), str(OUT), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=13, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    x = torch.randn(64, 16, 96) * 15
    a = s.run(None, {"input": x.numpy()})[0].reshape(-1); b = m(x).detach().numpy().reshape(-1)
    print("exported", OUT, OUT.stat().st_size, "bytes; onnx/torch max diff", float(np.abs(a - b).max()))
    (OUT.parent / "ensemble_members.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
