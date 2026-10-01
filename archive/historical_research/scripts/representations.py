"""Optional denoising and SCARF-style encoders, fitted without outcome labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import features  # Sets thread limits before importing numerical libraries.
import copy
import numpy as np
import torch
from features import CATEGORICAL as CATS, NUMERIC, QUESTIONS, SEED, clean
from sklearn.preprocessing import OneHotEncoder
from torch import nn



class AutoencoderInputs:
    """Identical preprocessing, fit only on labeled training X, for both AE variants."""
    def fit(self, d):
        numeric = d[NUMERIC].to_numpy(dtype=np.float32)
        self.mean = np.nanmean(numeric, axis=0)
        self.scale = np.nanstd(numeric, axis=0)
        self.mean = np.nan_to_num(self.mean)
        self.scale = np.where(self.scale > 1e-6, self.scale, 1.)
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32).fit(d[CATS])
        self.cat_sizes = [len(c) for c in self.onehot.categories_]
        self.cat_ranges = np.cumsum([0] + self.cat_sizes).tolist()
        self.groups = []
        # Average each modality's observed reconstruction errors before averaging modalities.
        # The loss balances blocks rather than weighting them by their column counts.
        groups = [
            ['age', 'tenure_months', 'wellness_optin', 'invalid_active_minutes'],
            [c for c in NUMERIC if c.startswith(('ocean_', 'engagement_'))] + ['burnout_composite'],
            [c for c in NUMERIC if c.startswith('wearable_')],
            ['workload_index', 'team_size', 'manager_support_score'],
        ]
        self.groups = [[NUMERIC.index(c) for c in group] for group in groups]
        return self

    def transform(self, d):
        raw = d[NUMERIC].to_numpy(dtype=np.float32)
        observed = np.isfinite(raw).astype(np.float32)
        numeric = np.nan_to_num((raw - self.mean) / self.scale, nan=0.)
        # Protect squared reconstruction loss from extreme magnitudes; raw LightGBM keeps originals.
        numeric = np.clip(numeric, -5, 5).astype(np.float32)
        cat = self.onehot.transform(d[CATS]).astype(np.float32)
        return numeric, observed, cat

class DenoisingAutoencoder(nn.Module):
    def __init__(self, numeric_count, cat_sizes, latent):
        super().__init__()
        self.numeric_count = numeric_count
        self.cat_sizes = cat_sizes
        output = numeric_count + sum(cat_sizes)
        self.encoder = nn.Sequential(nn.Linear(numeric_count * 2 + sum(cat_sizes), 64),
                                     nn.ReLU(), nn.Linear(64, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, output))

    def forward(self, numeric, observed, categorical):
        z = self.encoder(torch.cat([numeric, observed, categorical], dim=1))
        return z, self.decoder(z)

def reconstruction_loss(pred, numeric, observed, cat, groups, ranges):
    n = numeric.shape[1]
    errors = (pred[:, :n] - numeric).square() * observed
    losses = [errors[:, g].sum() / observed[:, g].sum().clamp_min(1.) for g in groups]
    category_losses = []
    for start, end in zip(ranges[:-1], ranges[1:]):
        valid = cat[:, start:end].sum(dim=1) > 0
        if valid.any():
            category_losses.append(nn.functional.cross_entropy(
                pred[valid, n+start:n+end], cat[valid, start:end].argmax(dim=1)))
    if category_losses:
        losses.append(torch.stack(category_losses).mean())
    return torch.stack(losses).mean()

class LearnedEncoder:
    def __init__(self, inputs, latent=16, seed=SEED):
        self.inputs = inputs
        self.latent = latent
        self.seed = seed

    def fit(self, frame, updates=2000, batch_size=512):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        self.net = DenoisingAutoencoder(len(NUMERIC), self.inputs.cat_sizes, self.latent)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        arrays = [torch.from_numpy(a) for a in self.inputs.transform(frame)]
        generator = torch.Generator().manual_seed(self.seed)
        history = []
        self.net.train()
        for step in range(1, updates + 1):
            idx = torch.randint(len(frame), (min(batch_size, len(frame)),), generator=generator)
            numeric, observed, categorical = [a[idx] for a in arrays]
            keep = (torch.rand(observed.shape, generator=generator) >= .15).float()
            noised_observed = observed * keep
            noised_cat = categorical.clone()
            for a, b in zip(self.inputs.cat_ranges[:-1], self.inputs.cat_ranges[1:]):
                visible = (torch.rand((len(idx), 1), generator=generator) >= .1).float()
                noised_cat[:, a:b] *= visible
            _, pred = self.net(numeric * noised_observed, noised_observed, noised_cat)
            loss = reconstruction_loss(pred, numeric, observed, categorical,
                                       self.inputs.groups, self.inputs.cat_ranges)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0:
                history.append({'step': step, 'batch_loss': float(loss.detach())})
        self.net.eval()
        self.history = history
        self.training_rows = len(frame)
        return self

    def transform(self, frame):
        self.net.eval()
        arrays = self.inputs.transform(frame)
        output = []
        with torch.no_grad():
            for start in range(0, len(frame), 4096):
                tensors = [torch.from_numpy(a[start:start+4096]) for a in arrays]
                z, _ = self.net(*tensors)
                output.append(z.numpy())
        return np.concatenate(output) if output else np.empty((0, self.latent))

class AugmentedRepresentation:
    def __init__(self, base, encoder):
        self.base = base
        self.encoder = encoder

    def transform(self, frame):
        # Keep original features: the downstream predictor is not forced through the bottleneck.
        frame = clean(frame)
        return np.column_stack([self.base.transform(frame), self.encoder.transform(frame)])


PAIRS = [(1, 19), (2, 20), (9, 17), (10, 18)]
PERSON_COLUMNS = ['age', 'tenure_months'] + [f'ocean_{c}' for c in 'ENACO'] + QUESTIONS + ['burnout_composite']
STEPS = [200]

def unique_people(frame, columns):
    # IDs select independent profiles, never numerical model inputs.
    if frame.groupby('person_id')[columns].nunique(dropna=False).to_numpy().max() > 1:
        raise ValueError('Person features vary between events; person-level representation is invalid.')
    return frame.sort_values('person_id').drop_duplicates('person_id').reset_index(drop=True)


class PersonInputs:
    def fit(self, frame):
        source = unique_people(frame,PERSON_COLUMNS); x=source[PERSON_COLUMNS].to_numpy(float)
        self.mean = np.nanmean(x,axis=0); self.scale = np.maximum(np.nanstd(x,axis=0),1e-6)
        self.mean = np.nan_to_num(self.mean); self.scale = np.nan_to_num(self.scale,nan=1.)
        self.groups = [[j] for j in range(len(PERSON_COLUMNS))]
        for a,b in PAIRS:
            ia,ib=[PERSON_COLUMNS.index(f'engagement_q{k:02d}') for k in (a,b)]
            self.groups = [g for g in self.groups if g not in [[ia],[ib]]] + [[ia,ib]]
        return self

    def transform(self, frame):
        values=frame[PERSON_COLUMNS].to_numpy(float); mask=np.isfinite(values).astype(np.float32)
        x=np.clip(np.nan_to_num((values-self.mean)/self.scale,nan=0.),-5,5).astype(np.float32)
        return np.concatenate([x,mask],axis=1)


class ContrastiveNet(nn.Module):
    def __init__(self,n):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(2*n,64),nn.ReLU(),nn.Linear(64,8))
        self.projection=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,8))
    def forward(self,x): return nn.functional.normalize(self.projection(self.encoder(x)),dim=1)


def corrupt_profiles(batch, source, groups, generator, anchors=None):
    n=batch.shape[1]//2; result=batch.clone()
    for group in groups:
        replace=torch.rand(len(batch),generator=generator)<.3
        if anchors is None:
            donors=torch.randint(len(source),(len(batch),),generator=generator)
        else:
            donors=(anchors+torch.randint(1,len(source),(len(batch),),generator=generator))%len(source)
        rows=torch.nonzero(replace,as_tuple=True)[0]; columns=torch.tensor(group+[j+n for j in group])
        result[rows[:,None],columns[None,:]]=source[donors[rows,None],columns[None,:]]
    return result


def info_nce(first,second,temperature=.2):
    n=len(first); both=torch.cat([first,second],dim=0); logits=both@both.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,dtype=torch.bool),-torch.inf)
    labels=(torch.arange(2*n)+n)%(2*n)
    return nn.functional.cross_entropy(logits,labels)


class ContrastiveEncoder:
    def __init__(self, inputs, seed): self.inputs,self.seed=inputs,seed
    def fit(self, source, diagnostic=False):
        source=unique_people(source,PERSON_COLUMNS); self.people=len(source)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(self.seed)
        self.net=ContrastiveNet(len(PERSON_COLUMNS)); optimizer=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.001)
        values=torch.from_numpy(self.inputs.transform(source)); generator=torch.Generator().manual_seed(self.seed+19)
        self.states={};self.history=[];steps=STEPS+[400] if diagnostic else STEPS
        for step in range(1,max(steps)+1):
            rows=torch.randperm(len(values),generator=generator)[:min(128,len(values))]
            batch=values[rows]; corrupted=corrupt_profiles(batch,values,self.inputs.groups,generator,rows)
            loss=info_nce(self.net(batch),self.net(corrupted));optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step in steps:
                self.states[step]=copy.deepcopy(self.net.state_dict()); self.history.append(dict(step=step,batch_loss=float(loss.detach())))
        return self
    def checkpoint(self,step):
        result=ContrastiveEncoder(self.inputs,self.seed);result.people=self.people
        result.net=ContrastiveNet(len(PERSON_COLUMNS));result.net.load_state_dict(self.states[step]);result.net.eval();result.step=step
        return result
    def transform(self,frame):
        self.net.eval();x=self.inputs.transform(frame);parts=[]
        with torch.inference_mode():
            for start in range(0,len(x),4096): parts.append(self.net.encoder(torch.from_numpy(x[start:start+4096])).numpy())
        return np.concatenate(parts).astype(float)
