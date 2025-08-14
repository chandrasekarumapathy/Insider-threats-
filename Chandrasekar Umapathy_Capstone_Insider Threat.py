# cert_insider_threat_colab_full.py
# Google Colab–compatible pipeline for Insider-Threat anomaly detection
# Datasets supported:
#  - CERT (v6) via big-archive link OR manual upload (extracts logon.csv, file.csv, device.csv)
#  - LANL Authentication dataset via direct URL OR manual upload (auth.csv or auth.txt[.gz])
#  - Enron Email dataset via prepared CSV URL OR TGZ raw corpus URL OR manual upload
#
# Steps:
# 0) Setup
# 1) CERT fetch & selective extract
# 2) LANL fetch/prepare
# 3) Enron fetch/prepare
# 4) Load & Sessionize
# 5) Scale & Split
# 6) Autoencoder (PyTorch)
# 7) Optional IsolationForest baseline + fusion
# 8) Optional metrics & plots

# ------------------------
# Step 0 — Setup
# ------------------------
# In Colab, first run:
# !pip -q install pandas numpy scikit-learn matplotlib torch --extra-index-url https://download.pytorch.org/whl/cpu

import os, random, json, zipfile, tarfile, io, shutil, glob, gzip, bz2, re
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = Path('/content/data'); BASE.mkdir(parents=True, exist_ok=True)
CERT_DIR  = BASE/'CERT';  CERT_DIR.mkdir(exist_ok=True)
LANL_DIR  = BASE/'LANL';  LANL_DIR.mkdir(exist_ok=True)
ENRON_DIR = BASE/'ENRON'; ENRON_DIR.mkdir(exist_ok=True)

# Controls
USE_CERT  = True
USE_LANL  = True
USE_ENRON = True

ROW_LIMIT_CERT  = 1_000_000    # None for full
ROW_LIMIT_LANL  = 2_000_000
ROW_LIMIT_ENRON = 500_000      # for parsing TGZ to CSV

VAL_PCTL = 95  # AE threshold percentile on validation

print('Dirs ready:', CERT_DIR, LANL_DIR, ENRON_DIR)

# ------------------------
# Utilities — download & extract
# ------------------------
import requests

def stream_download(url: str, dest: Path, chunk: int = 1<<20, timeout: int = 300):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total = int(r.headers.get('Content-Length', 0)); done = 0
    with open(dest, 'wb') as f:
        for ch in r.iter_content(chunk):
            if ch:
                f.write(ch); done += len(ch)
                if total:
                    print(f'\r{dest.name}: {done/1_048_576:.1f}/{total/1_048_576:.1f} MB', end='')
    print()

def selective_extract_any(archive_path: Path, wanted: set, target_dir: Path) -> List[Path]:
    # Extract only basenames in 'wanted' from zip/tar.* or single-compressed files.
    extracted = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as z:
                for name in z.namelist():
                    base = Path(name).name
                    if (not wanted or base in wanted) and not (target_dir/base).exists():
                        print('Extract:', base, 'from', archive_path.name)
                        with z.open(name) as src, open(target_dir/base, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        extracted.append(target_dir/base)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as t:
                for m in t.getmembers():
                    if m.isfile():
                        base = Path(m.name).name
                        if (not wanted or base in wanted) and not (target_dir/base).exists():
                            print('Extract:', base, 'from', archive_path.name)
                            with t.extractfile(m) as src, open(target_dir/base, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                            extracted.append(target_dir/base)
        else:
            lname = archive_path.name.lower()
            if lname.endswith('.gz') and not lname.endswith('.tar.gz'):
                base = Path(archive_path.stem).name
                if (not wanted or base in wanted) and not (target_dir/base).exists():
                    with gzip.open(archive_path, 'rb') as src, open(target_dir/base, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(target_dir/base)
            elif lname.endswith('.bz2') and not lname.endswith('.tar.bz2'):
                base = Path(archive_path.stem).name
                if (not wanted or base in wanted) and not (target_dir/base).exists():
                    with bz2.open(archive_path, 'rb') as src, open(target_dir/base, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(target_dir/base)
    except Exception as e:
        print('Selective extract error:', e)
    return extracted

# ------------------------
# Step 1 — CERT (big link OR upload)
# ------------------------
BIG_ARCHIVE_LINK_CERT = "https://kilthub.cmu.edu/ndownloader/files/24857828"  # "" to upload
WANTED_CERT = {"logon.csv","file.csv","device.csv"}

def get_cert_data():
    archives = []
    if BIG_ARCHIVE_LINK_CERT:
        big = CERT_DIR / Path(BIG_ARCHIVE_LINK_CERT).name
        if not big.exists():
            print('Downloading CERT big archive…')
            stream_download(BIG_ARCHIVE_LINK_CERT, big)
        else:
            print('CERT: archive already present:', big.name)
        archives = [big]
    else:
        try:
            from google.colab import files
            print('Upload a CERT archive (zip/tar.gz/tgz)…')
            uploaded = files.upload()
            for name, data in uploaded.items():
                p = CERT_DIR / name
                with open(p,'wb') as f: f.write(data)
                archives.append(p)
                print('Uploaded:', name)
        except Exception as e:
            print('Upload failed:', e)
    x = []
    for arc in archives:
        x += selective_extract_any(arc, WANTED_CERT, CERT_DIR)
    if not any((CERT_DIR/w).exists() for w in WANTED_CERT):
        TMP = CERT_DIR/'_tmp'; TMP.mkdir(exist_ok=True)
        for arc in archives:
            try:
                if zipfile.is_zipfile(arc):
                    with zipfile.ZipFile(arc) as z: z.extractall(TMP)
                elif tarfile.is_tarfile(arc):
                    with tarfile.open(arc,'r:*') as t: t.extractall(TMP)
            except Exception as e:
                print('Top-level extract skip:', arc.name, '->', e)
        inner = [p for p in TMP.rglob('*') if p.is_file() and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p) or p.suffix in ['.gz','.bz2'])]
        for ia in inner:
            x += selective_extract_any(ia, WANTED_CERT, CERT_DIR)
        shutil.rmtree(TMP, ignore_errors=True)
    print('CERT contents:', [p.name for p in CERT_DIR.iterdir()])
    missing = [w for w in WANTED_CERT if not (CERT_DIR/w).exists()]
    print('CERT missing:', missing if missing else 'None — all found!')

# ------------------------
# Step 2 — LANL authentication (URL OR upload)
# ------------------------
LANL_AUTH_URL = ""  # set to direct CSV/TXT(.gz) if you have; else upload in Colab

def prepare_lanl():
    if LANL_AUTH_URL:
        target = LANL_DIR / Path(LANL_AUTH_URL).name
        if not target.exists():
            print('Downloading LANL auth…')
            stream_download(LANL_AUTH_URL, target)
        # try to extract if archive
        if target.suffix in ('.gz', '.bz2', '.zip', '.tgz', '.tar', '.tar.gz'):
            extracted = selective_extract_any(target, set(), LANL_DIR)
            print('LANL extracted (best-effort). Ensure auth.csv exists or upload.')
    else:
        try:
            from google.colab import files
            print('Upload LANL auth file (auth.csv or auth.txt[.gz])…')
            uploaded = files.upload()
            for name, data in uploaded.items():
                p = LANL_DIR / name
                with open(p, 'wb') as f: f.write(data)
                print('Uploaded:', name)
        except Exception as e:
            print('LANL upload skipped:', e)

# ------------------------
# Step 3 — Enron (CSV URL OR TGZ OR upload)
# ------------------------
ENRON_CSV_URL = ""  # prepared CSV with columns from,to,subject,body,date
ENRON_TGZ_URL = ""  # raw corpus (e.g., https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz)

def parse_enron_tgz_to_csv(tgz_path: Path, out_csv: Path, row_limit: Optional[int] = 500000):
    import tarfile
    rows = []
    with tarfile.open(tgz_path, 'r:*') as t:
        members = [m for m in t.getmembers() if m.isfile() and (m.name.endswith('.txt') or '/sent/' in m.name or '/inbox/' in m.name)]
        for i, m in enumerate(members):
            if row_limit and len(rows) >= row_limit: break
            try:
                f = t.extractfile(m)
                if not f: continue
                raw = f.read().decode('latin1', errors='ignore')
                frm = re.search(r'^From:\s*(.+)', raw, re.M)
                to  = re.search(r'^To:\s*(.+)', raw, re.M)
                sub = re.search(r'^Subject:\s*(.+)', raw, re.M)
                dat = re.search(r'^Date:\s*(.+)', raw, re.M)
                body_split = re.split(r'\r?\n\r?\n', raw, maxsplit=1)
                body = body_split[1] if len(body_split) > 1 else ''
                rows.append({'from': frm.group(1).strip() if frm else '',
                             'to': to.group(1).strip() if to else '',
                             'subject': sub.group(1).strip() if sub else '',
                             'body': body.strip(),
                             'date': dat.group(1).strip() if dat else ''})
            except Exception:
                continue
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

def get_enron_data():
    if ENRON_CSV_URL:
        target = ENRON_DIR / Path(ENRON_CSV_URL).name
        if not target.exists():
            print('Downloading Enron CSV…')
            stream_download(ENRON_CSV_URL, target)
        out = ENRON_DIR/'emails.csv'
        try:
            df = pd.read_csv(target, low_memory=False)
            keep = [c for c in ['from','to','subject','body','date'] if c in df.columns]
            if len(keep) >= 3:
                df[keep].to_csv(out, index=False)
                print('Saved standardized emails.csv with columns:', keep)
            else:
                print('CSV missing expected columns; copying as emails.csv')
                target.rename(out)
        except Exception:
            target.rename(out)
        print('Enron emails at:', out)
    elif ENRON_TGZ_URL:
        tgz = ENRON_DIR / Path(ENRON_TGZ_URL).name
        if not tgz.exists():
            print('Downloading Enron TGZ…')
            stream_download(ENRON_TGZ_URL, tgz)
        out = ENRON_DIR/'emails.csv'
        print('Parsing Enron TGZ to CSV (may take time)…')
        parse_enron_tgz_to_csv(tgz, out, row_limit=ROW_LIMIT_ENRON)
        print('Enron parsed to:', out)
    else:
        try:
            from google.colab import files
            print('Upload Enron emails CSV (from,to,subject,body,date) OR TGZ corpus…')
            uploaded = files.upload()
            for name, data in uploaded.items():
                p = ENRON_DIR / name
                with open(p,'wb') as f: f.write(data)
                print('Uploaded:', name)
                if p.suffix == '.tgz' or p.suffixes[-2:] == ['.tar','.gz']:
                    out = ENRON_DIR/'emails.csv'
                    print('Parsing uploaded TGZ to emails.csv …')
                    parse_enron_tgz_to_csv(p, out, row_limit=ROW_LIMIT_ENRON)
        except Exception as e:
            print('Enron upload skipped:', e)

if USE_CERT:  get_cert_data()
if USE_LANL:  prepare_lanl()
if USE_ENRON: get_enron_data()

# ------------------------
# Step 4 — Load dataframes
# ------------------------
def read_csv(path, parse_dates=None, nrows=None, usecols=None):
    p = Path(path)
    if not p.exists():
        print('Missing:', p.name)
        return None
    return pd.read_csv(p, low_memory=False, parse_dates=parse_dates, nrows=nrows, usecols=usecols)

logon  = read_csv(CERT_DIR/'logon.csv',  parse_dates=['time'], nrows=ROW_LIMIT_CERT) if USE_CERT else None
files  = read_csv(CERT_DIR/'file.csv',   parse_dates=['time'], nrows=ROW_LIMIT_CERT) if USE_CERT else None
device = read_csv(CERT_DIR/'device.csv', parse_dates=['time'], nrows=ROW_LIMIT_CERT) if USE_CERT else None

auth_path = None
if USE_LANL:
    cand = list(LANL_DIR.glob('auth.csv')) + list(LANL_DIR.glob('*.csv'))
    if not cand:
        txts = list(LANL_DIR.glob('auth.txt')) + list(LANL_DIR.glob('auth.txt.gz'))
        if txts:
            p = txts[0]
            if p.suffix == '.gz':
                with gzip.open(p, 'rt', errors='ignore') as f, open(LANL_DIR/'auth.csv','w') as out:
                    out.write(f.read()); auth_path = LANL_DIR/'auth.csv'
            else:
                auth_path = p
    else:
        auth_path = cand[0]

auth = None
if USE_LANL and auth_path:
    try:
        auth = pd.read_csv(auth_path, nrows=ROW_LIMIT_LANL, low_memory=False, parse_dates=['time'])
    except Exception:
        auth = pd.read_csv(auth_path, nrows=ROW_LIMIT_LANL, low_memory=False)
        if 'time' in auth.columns:
            auth['time'] = pd.to_datetime(auth['time'], errors='coerce')

emails = read_csv(ENRON_DIR/'emails.csv', parse_dates=['date'], nrows=ROW_LIMIT_ENRON) if USE_ENRON else None

print('Shapes — CERT:', *(x.shape if x is not None else None for x in [logon, files, device]), '| LANL:', None if auth is None else auth.shape, '| Enron:', None if emails is None else emails.shape)

# ------------------------
# Step 5 — Sessionization
# ------------------------
def sessionize_cert(logon, files, device):
    if any(df is None for df in [logon,files,device]): return None
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
    for src in ['user','user_name','source_user','account']:
        if src in a.columns:
            a.rename(columns={src:'user'}, inplace=True); break
    for sh in ['src_host','source_host','src','computer']:
        if sh in a.columns:
            a.rename(columns={sh:'src_host'}, inplace=True); break
    for dh in ['dst_host','dest_host','destination_host','target_host']:
        if dh in a.columns:
            a.rename(columns={dh:'dst_host'}, inplace=True); break
    if 'success' not in a.columns and 'status' in a.columns:
        a['success'] = a['status'].astype(str).str.lower().isin(['success','ok','true','1']).astype(int)
    a['time'] = pd.to_datetime(a['time'], errors='coerce')
    a['week'] = a['time'].dt.to_period('W').dt.start_time
    g = a.groupby(['user','week'], sort=False).agg(
        auth_events=('success','count'),
        auth_success=('success','sum'),
        uniq_src=('src_host','nunique'),
        uniq_dst=('dst_host','nunique'),
        night_auth=('time', lambda c: ((c.dt.hour<7)|(c.dt.hour>20)).sum())
    ).reset_index()
    g['success_rate'] = (g['auth_success']/g['auth_events']).fillna(0)
    return g

def sessionize_enron(emails):
    if emails is None: return None
    e = emails.copy()
    for c in ['from','sender','from_address']:
        if c in e.columns: e.rename(columns={c:'from'}, inplace=True); break
    for c in ['to','recipients','to_address']:
        if c in e.columns: e.rename(columns={c:'to'}, inplace=True); break
    for c in ['subject','Subject']:
        if c in e.columns: e.rename(columns={c:'subject'}, inplace=True); break
    for c in ['date','Date','datetime']:
        if c in e.columns: e.rename(columns={c:'date'}, inplace=True); break
    e['date'] = pd.to_datetime(e['date'], errors='coerce')
    e['user'] = e['from'].astype(str).str.extract(r'(^[^@\s>]+)')
    e['week'] = e['date'].dt.to_period('W').dt.start_time
    e['len_body'] = e.get('body','').astype(str).str.len() if 'body' in e.columns else 0
    kw = r'\b(resign|quit|leaving|angry|frustrat|complain)\b'
    e['kw_hits'] = (e.get('body','').astype(str).str.lower().str.contains(kw, regex=True)).astype(int) if 'body' in e.columns else 0
    g = e.groupby(['user','week'], sort=False).agg(
        emails=('subject','count'),
        mean_len=('len_body','mean'),
        kw_hits=('kw_hits','sum')
    ).reset_index()
    return g

cert_s = sessionize_cert(logon, files, device) if USE_CERT else None
lanl_s = sessionize_lanl(auth) if USE_LANL else None
enron_s= sessionize_enron(emails) if USE_ENRON else None

print('Sessionized — CERT:', None if cert_s is None else cert_s.shape,
      'LANL:', None if lanl_s is None else lanl_s.shape,
      'ENRON:', None if enron_s is None else enron_s.shape)

# ------------------------
# Step 6 — Scale & split
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
    Xc, sc_c = to_X(cert_s, cert_cols); tr,va,te = split_idx(len(Xc)); blocks['CERT'] = (Xc[tr], Xc[va], Xc[te])
if lanl_s is not None and set(lanl_cols).issubset(lanl_s.columns):
    Xl, sc_l = to_X(lanl_s, lanl_cols); tr,va,te = split_idx(len(Xl)); blocks['LANL'] = (Xl[tr], Xl[va], Xl[te])
if enron_s is not None and set(enron_cols).issubset(enron_s.columns):
    Xe, sc_e = to_X(enron_s, enron_cols); tr,va,te = split_idx(len(Xe)); blocks['ENRON'] = (Xe[tr], Xe[va], Xe[te])

print('Prepared blocks:', {k: tuple(x.shape for x in v) for k,v in blocks.items()})

# ------------------------
# Step 7 — Autoencoder
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
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        if ep % 5 == 0: print(f'epoch {ep:02d}  train={tr_loss:.5f}  val={va_loss:.5f}')
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
    print(f'[{name}] training AE d={Xtr.shape[1]}  ntr={len(Xtr)} nva={len(Xva)}')
    m, dev = train_ae(Xtr, Xva, Xtr.shape[1], epochs=20)
    e_va = recon_err(m, Xva, dev)
    thr  = float(np.percentile(e_va, VAL_PCTL))
    e_te = recon_err(m, Xte, dev)
    flags = (e_te > thr).astype(int)
    scores[name] = dict(thr=thr, err_val=e_va, err_test=e_te, flags=flags)
print({k: (v['err_test'].shape, v['thr']) for k,v in scores.items()})

# ------------------------
# Step 8 — Optional IsolationForest + fusion
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
print('Fused anomaly rate:', None if fused_flags is None else float(fused_flags.mean()))

# ------------------------
# Step 9 — Optional metrics & plots
# ------------------------
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# y_true_cert = ...  # align labels to CERT test indices
# prec, rec, f1, _ = precision_recall_fscore_support(y_true_cert, scores['CERT']['flags'], average='binary', zero_division=0)
# auc = roc_auc_score(y_true_cert, scores['CERT']['err_test'])
# print(f"CERT  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
