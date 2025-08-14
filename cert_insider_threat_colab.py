# cert_insider_threat_colab.py
# Google Colab–compatible pipeline for CERT (v6) insider-threat anomaly detection
# - Step 0: Setup
# - Step 1: Download or Upload big archive and selective extraction (logon.csv, file.csv, device.csv)
# - Step 2: Load CSVs
# - Step 3: Sessionize weekly per user
# - Step 4: Scale & Split
# - Step 5: Autoencoder training (PyTorch)
# - Step 6: Optional IsolationForest baseline + fusion
# - Step 7: Optional metrics hooks (if labels available)
# - Step 8: Optional visualization

# ------------------------
# Step 0 — Setup
# ------------------------
# In Colab, run:
# !pip -q install pandas numpy scikit-learn matplotlib torch --extra-index-url https://download.pytorch.org/whl/cpu

import os, random, json, zipfile, tarfile, io, shutil, glob, gzip, bz2
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = Path("/content/data"); BASE.mkdir(parents=True, exist_ok=True)
CERT_DIR = BASE/"CERT"; CERT_DIR.mkdir(exist_ok=True)
LANL_DIR = BASE/"LANL"; LANL_DIR.mkdir(exist_ok=True)
ENRON_DIR = BASE/"ENRON"; ENRON_DIR.mkdir(exist_ok=True)

USE_CERT  = True
USE_LANL  = False
USE_ENRON = False

ROW_LIMIT_CERT  = 1_000_000
ROW_LIMIT_LANL  = 2_000_000
ROW_LIMIT_ENRON = 500_000

VAL_PCTL = 95  # threshold percentile for AE validation errors

print("Dirs ready:", CERT_DIR, LANL_DIR, ENRON_DIR)

# ------------------------
# Step 1 — Get CERT data (one big link OR upload)
# ------------------------
# Example big archive link (replace as needed; leave empty to upload manually in Colab)
BIG_ARCHIVE_LINK = "https://kilthub.cmu.edu/ndownloader/files/24857828"  # or "" to upload

WANTED = {"logon.csv", "file.csv", "device.csv"}

def stream_download(url: str, dest: Path, chunk: int = 1<<20):
    import requests
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0)); done = 0
    with open(dest, "wb") as f:
        for ch in r.iter_content(chunk):
            if ch:
                f.write(ch); done += len(ch)
                if total:
                    print(f"\r{dest.name}: {done/1_048_576:.1f}/{total/1_048_576:.1f} MB", end="")
    print()

def selective_extract_any(archive_path: Path, wanted: set, target_dir: Path):
    """Extract only specific basenames from zip/tar.* or single-compressed files."""
    extracted = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as z:
                for name in z.namelist():
                    base = Path(name).name
                    if base in wanted and not (target_dir/base).exists():
                        print("Extract:", base, "from", archive_path.name)
                        with z.open(name) as src, open(target_dir/base, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted.append(target_dir/base)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as t:
                for m in t.getmembers():
                    if m.isfile():
                        base = Path(m.name).name
                        if base in wanted and not (target_dir/base).exists():
                            print("Extract:", base, "from", archive_path.name)
                            with t.extractfile(m) as src, open(target_dir/base, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            extracted.append(target_dir/base)
        else:
            lname = archive_path.name.lower()
            if lname.endswith(".gz") and not lname.endswith(".tar.gz"):
                base = Path(archive_path.stem).name
                if base in wanted and not (target_dir/base).exists():
                    with gzip.open(archive_path, "rb") as src, open(target_dir/base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(target_dir/base)
            elif lname.endswith(".bz2") and not lname.endswith(".tar.bz2"):
                base = Path(archive_path.stem).name
                if base in wanted and not (target_dir/base).exists():
                    with bz2.open(archive_path, "rb") as src, open(target_dir/base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(target_dir/base)
    except Exception as e:
        print("Selective extract error:", e)
    return extracted

def get_cert_data():
    archives = []
    if BIG_ARCHIVE_LINK:
        big = CERT_DIR / Path(BIG_ARCHIVE_LINK).name
        if not big.exists():
            print("Downloading big archive…")
            stream_download(BIG_ARCHIVE_LINK, big)
        else:
            print("Archive already present:", big.name)
        archives = [big]
    else:
        try:
            from google.colab import files
            print("No link provided — please upload a CERT archive (zip / tar.gz / tgz).")
            uploaded = files.upload()
            for name, data in uploaded.items():
                p = CERT_DIR / name
                with open(p, "wb") as f: f.write(data)
                archives.append(p)
                print("Uploaded:", name)
        except Exception as e:
            print("Upload failed:", e)
    # Try direct extraction
    extracted = []
    for arc in archives:
        extracted += selective_extract_any(arc, WANTED, CERT_DIR)

    # If none found, expand one level and re-try extraction on inner archives
    if not any((CERT_DIR/w).exists() for w in WANTED):
        TMP = CERT_DIR/"_tmp"; TMP.mkdir(exist_ok=True)
        for arc in archives:
            try:
                if zipfile.is_zipfile(arc):
                    with zipfile.ZipFile(arc) as z: z.extractall(TMP)
                elif tarfile.is_tarfile(arc):
                    with tarfile.open(arc, "r:*") as t: t.extractall(TMP)
            except Exception as e:
                print("Top-level extract skip:", arc.name, "->", e)
        inner_archives = [p for p in TMP.rglob("*") if p.is_file() and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p) or p.suffix in [".gz",".bz2"])]
        for inner in inner_archives:
            extracted += selective_extract_any(inner, WANTED, CERT_DIR)
        shutil.rmtree(TMP, ignore_errors=True)

    print("CERT_DIR contains:", [p.name for p in CERT_DIR.iterdir()])
    missing = [w for w in WANTED if not (CERT_DIR/w).exists()]
    print("Missing:", missing if missing else "None — all found!")

if USE_CERT:
    get_cert_data()

# ------------------------
# Step 2 — Load CSVs
# ------------------------
def read_csv(path, parse_dates=None, nrows=None, usecols=None):
    p = Path(path)
    if not p.exists():
        print("Missing:", p.name)
        return None
    return pd.read_csv(p, low_memory=False, parse_dates=parse_dates, nrows=nrows, usecols=usecols)

logon  = read_csv(CERT_DIR/"logon.csv",  parse_dates=["time"], nrows=ROW_LIMIT_CERT) if USE_CERT else None
files  = read_csv(CERT_DIR/"file.csv",   parse_dates=["time"], nrows=ROW_LIMIT_CERT) if USE_CERT else None
device = read_csv(CERT_DIR/"device.csv", parse_dates=["time"], nrows=ROW_LIMIT_CERT) if USE_CERT else None

print("CERT shapes:", *(x.shape if x is not None else None for x in [logon, files, device]))

auth   = read_csv(LANL_DIR/"auth.csv", parse_dates=["time"], nrows=ROW_LIMIT_LANL) if USE_LANL else None
emails = read_csv(ENRON_DIR/"emails.csv", parse_dates=["date"], nrows=ROW_LIMIT_ENRON) if USE_ENRON else None
print("LANL:", None if auth is None else auth.shape, " | Enron:", None if emails is None else emails.shape)

# ------------------------
# Step 3 — Sessionize (weekly per user)
# ------------------------
def sessionize_cert(logon, files, device):
    if any(df is None for df in [logon, files, device]):
        return None
    def rn(df, m):
        m = {k:v for k,v in m.items() if k in df.columns and v not in df.columns}
        return df.rename(columns=m) if m else df
    logon  = rn(logon,  {'event':'activity','computer':'pc','id':'user','date':'time'})
    files  = rn(files,  {'activity':'operation','date':'time'})
    device = rn(device, {'activity':'action','date':'time'})

    for df in (logon, files, device):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    logon['week']  = logon['time'].dt.to_period('W').dt.start_time
    files['week']  = files['time'].dt.to_period('W').dt.start_time
    device['week'] = device['time'].dt.to_period('W').dt.start_time

    g1 = logon.groupby(['user','week'], sort=False).agg(
        logons=('activity', lambda s: (s.astype(str).str.lower().eq('logon')).sum()),
        logoffs=('activity', lambda s: (s.astype(str).str.lower().eq('logoff')).sum()),
        unique_pcs=('pc','nunique'),
        offhour_logons=('time', lambda c: ((c.dt.hour<8)|(c.dt.hour>19)).sum())
    ).reset_index()

    g2 = files.groupby(['user','week'], sort=False).agg(
        file_ops=('operation','count'),
        file_writes=('operation', lambda s: (s.astype(str).str.lower().eq('write')).sum()),
        bytes_total=('bytes','sum') if 'bytes' in files.columns else ('operation','count')
    ).reset_index()

    g3 = device.groupby(['user','week'], sort=False).agg(
        usb_events=('action','count'),
        usb_mounts=('action', lambda s: (s.astype(str).str.lower().eq('mount')).sum())
    ).reset_index()

    out = g1.merge(g2, on=['user','week'], how='left').merge(g3, on=['user','week'], how='left').fillna(0)
    return out

def sessionize_lanl(auth):
    if auth is None: return None
    a = auth.copy()
    a['time'] = pd.to_datetime(a['time'], errors='coerce')
    a['week'] = a['time'].dt.to_period('W').dt.start_time
    g = a.groupby(['user','week'], sort=False).agg(
        auth_events=('success','count'),
        auth_success=('success','sum'),
        uniq_src=('src_host','nunique'),
        uniq_dst=('dst_host','nunique'),
        night_auth=('time', lambda c: ((c.dt.hour<7)|(c.dt.hour>20)).sum())
    ).reset_index()
    g['success_rate'] = (g['auth_success'] / g['auth_events']).fillna(0)
    return g

def sessionize_enron(emails):
    if emails is None: return None
    e = emails.copy()
    e['user'] = e['from'].str.extract(r'(^[^@]+)')
    e['week'] = e['date'].dt.to_period('W').dt.start_time
    e['len_body'] = e['body'].fillna('').astype(str).str.len()
    kw = r'\b(resign|quit|leaving|angry|frustrat|complain)\b'
    e['kw_hits'] = e['body'].fillna('').str.lower().str.contains(kw).astype(int)
    g = e.groupby(['user','week'], sort=False).agg(
        emails=('body','count'),
        mean_len=('len_body','mean'),
        kw_hits=('kw_hits','sum')
    ).reset_index()
    return g

cert_s = sessionize_cert(logon, files, device) if USE_CERT else None
lanl_s = sessionize_lanl(auth) if USE_LANL else None
enron_s= sessionize_enron(emails) if USE_ENRON else None

print("Sessionized:", *(x.shape if x is not None else None for x in [cert_s, lanl_s, enron_s]))

# ------------------------
# Step 4 — Scale & split
# ------------------------
from sklearn.preprocessing import StandardScaler

def to_X(df, cols):
    X = df[cols].astype('float32').values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype('float32')
    return Xs, scaler

def split_idx(n, train=0.8, val=0.1):
    idx = np.arange(n); np.random.shuffle(idx)
    ntr = int(n*train); nva = int(n*val)
    return idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

cert_cols  = ['logons','logoffs','unique_pcs','offhour_logons','file_ops','file_writes','bytes_total','usb_events','usb_mounts']
lanl_cols  = ['auth_events','auth_success','uniq_src','uniq_dst','night_auth','success_rate']
enron_cols = ['emails','mean_len','kw_hits']

blocks = {}
if cert_s is not None and set(cert_cols).issubset(cert_s.columns):
    Xc, sc_c = to_X(cert_s, cert_cols); tr,va,te = split_idx(len(Xc))
    blocks['CERT'] = (Xc[tr], Xc[va], Xc[te])
if lanl_s is not None and set(lanl_cols).issubset(lanl_s.columns):
    Xl, sc_l = to_X(lanl_s, lanl_cols); tr,va,te = split_idx(len(Xl))
    blocks['LANL'] = (Xl[tr], Xl[va], Xl[te])
if enron_s is not None and set(enron_cols).issubset(enron_s.columns):
    Xe, sc_e = to_X(enron_s, enron_cols); tr,va,te = split_idx(len(Xe))
    blocks['ENRON'] = (Xe[tr], Xe[va], Xe[te])

print("Prepared blocks:", {k: tuple(x.shape for x in v) for k,v in blocks.items()})

# ------------------------
# Step 5 — Autoencoder (PyTorch)
# ------------------------
import torch, torch.nn as nn

class AE(nn.Module):
    def __init__(self, d, h=None, bottleneck=8):
        super().__init__()
        h = h or max(16, 2*d)
        self.enc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck,h), nn.ReLU(), nn.Linear(h,d))
    def forward(self, x): return self.dec(self.enc(x))

def train_ae(Xtr, Xva, d, epochs=20, lr=1e-3, bs=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(d).to(dev); opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def loader(X, sh=False):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(X))
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=sh, pin_memory=True)

    Ltr, Lva = loader(Xtr, True), loader(Xva, False)
    best = (1e9, None)
    for ep in range(epochs):
        model.train(); s=0
        for xb,yb in Ltr:
            xb,yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); out = model(xb); loss = loss_fn(out,yb)
            loss.backward(); opt.step(); s += loss.item()*xb.size(0)
        tr_loss = s/len(Ltr.dataset)

        model.eval(); s=0
        with torch.no_grad():
            for xb,yb in Lva:
                xb,yb = xb.to(dev), yb.to(dev)
                out = model(xb); loss = loss_fn(out,yb); s += loss.item()*xb.size(0)
        va_loss = s/len(Lva.dataset)
        if va_loss < best[0]: best = (va_loss, model.state_dict())
        if ep % 5 == 0: print(f"epoch {ep:02d}  train={tr_loss:.5f}  val={va_loss:.5f}")
    model.load_state_dict(best[1]); model.eval()
    return model, dev

def recon_err(model, X, dev, bs=4096):
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(dev)
            pred = model(xb).cpu().numpy()
            e = ((pred - X[i:i+bs])**2).mean(axis=1)
            errs.append(e)
    return np.concatenate(errs)

scores = {}
for name, (Xtr, Xva, Xte) in blocks.items():
    print(f"[{name}] training AE d={Xtr.shape[1]}  ntr={len(Xtr)} nva={len(Xva)}")
    m, dev = train_ae(Xtr, Xva, Xtr.shape[1], epochs=20)
    e_va = recon_err(m, Xva, dev)
    thr  = float(np.percentile(e_va, VAL_PCTL))
    e_te = recon_err(m, Xte, dev)
    flags = (e_te > thr).astype(int)
    scores[name] = dict(thr=thr, err_val=e_va, err_test=e_te, flags=flags)
print({ k: (v['err_test'].shape, v['thr']) for k,v in scores.items() })

# ------------------------
# Step 6 — (Optional) IsolationForest + fusion
# ------------------------
from sklearn.ensemble import IsolationForest

def iso_score(Xtr, Xte):
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iso.fit(Xtr)
    s = -iso.decision_function(Xte)
    s = (s - s.min())/(s.ptp()+1e-9)
    return s

for name, (Xtr, Xva, Xte) in blocks.items():
    s = iso_score(Xtr, Xte)
    scores[name]['iso'] = s

fused = None
for name in scores:
    f = scores[name]['flags']
    fused = f if fused is None else fused + f
fused_flags = (fused >= max(1, len(scores)//2 + 1)).astype(int) if fused is not None else None
print("Fused anomaly rate:", None if fused_flags is None else float(fused_flags.mean()))

# ------------------------
# Step 7 — (Optional) metrics hooks (if labels available)
# ------------------------
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# y_true_cert = ...  # align labels to CERT test subset
# prec, rec, f1, _ = precision_recall_fscore_support(y_true_cert, scores['CERT']['flags'], average='binary', zero_division=0)
# auc = roc_auc_score(y_true_cert, scores['CERT']['err_test'])
# print(f"CERT  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")

# ------------------------
# Step 8 — (Optional) quick visualization
# ------------------------
# import matplotlib.pyplot as plt
# def plot_hist(err, thr, title):
#     plt.figure(figsize=(5,3))
#     plt.hist(err, bins=60)
#     plt.axvline(thr, ls='--')
#     plt.title(title); plt.tight_layout(); plt.show()
# for name in scores:
#     plot_hist(scores[name]['err_test'], scores[name]['thr'], f"{name} AE Reconstruction Errors (test)")
