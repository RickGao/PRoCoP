"""Mechanics-only synthetic checks, not a CircuitNet experiment."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from model import FCN
from train import soft_ap_loss, weighted_mse

torch.set_num_threads(2)
torch.manual_seed(42)
x = torch.rand(2, 3, 32, 32)
y = torch.rand(2, 1, 32, 32)
model = FCN()
pred = model(x)
assert pred.shape == y.shape
assert torch.allclose(weighted_mse(pred,y,0,.5), (pred-y).square().mean())
weighted_mse(pred,y,3,.5).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
model.eval()
opt = torch.optim.Adam(model.parameters(), lr=.002)
target = torch.full_like(y, .2)
initial = (model(x)-target).square().mean().item()
for _ in range(20):
    opt.zero_grad(); loss = (model(x)-target).square().mean(); loss.backward(); opt.step()
final = (model(x)-target).square().mean().item()
assert final < initial, (initial,final)

# Positives at ranks 1 and 3 give AP (1 + 2/3) / 2 at the soft-bin centers.
logits = torch.logit(torch.tensor([2., 8., 19., 27.]) / 31).requires_grad_()
mask = torch.tensor([0., 1., 0., 1.])
ap_loss = soft_ap_loss(logits, mask, .5)
torch.testing.assert_close(ap_loss, torch.tensor(1/6), atol=1e-6, rtol=0)
ap_loss.backward()
assert torch.isfinite(logits.grad).all()
assert soft_ap_loss(logits, torch.zeros_like(mask), .5).item() == 0

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp); rows=[]
    rng = np.random.default_rng(42)
    for design in ['synthetic_A','synthetic_B','synthetic_C']:
        for i in range(2):
            f,l = f'{design}_{i}_x.npy', f'{design}_{i}_y.npy'
            a = rng.random((32,32,3), dtype=np.float32)
            np.save(root/f,a); np.save(root/l,a[:,:,:1]*.5)
            rows.append([design,f,l])
    with (root/'manifest.csv').open('w') as file:
        w=csv.writer(file); w.writerow(['design','feature','label']); w.writerows(rows)
    for alpha in [0,3]:
        subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),
                        '--manifest',str(root/'manifest.csv'),'--test-design','synthetic_A',
                        '--val-design','synthetic_B','--epochs','1','--alpha',str(alpha),
                        '--out',str(root/f'run_{alpha}')],check=True)
    spatial_run = root / 'spatial'
    subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')),
                    '--manifest', str(root / 'manifest.csv'), '--test-design', 'synthetic_A',
                    '--val-design', 'synthetic_B', '--model', 'unet-spatial', '--ap-loss-weight', '0.005',
                    '--device', 'cpu', '--epochs', '1', '--select-by', 'ap', '--no-progress',
                    '--out', str(spatial_run)], check=True)
    metrics = json.loads((spatial_run / 'test_metrics.json').read_text())
    assert metrics['ranking_score'] == 'hotspot_probability' and 0 <= metrics['ap'] <= 1
print(f'PASS: synthetic FCN and spatial training, AP loss and overfit ({initial:.4f} -> {final:.4f}).')
