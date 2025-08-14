# insider_threat_pipeline.py
# -----------------------------------------------------------------------------
# Supports: CERT (logon/file/device), LANL (auth), ENRON (emails)
# Models: Autoencoder (PyTorch) + Isolation Forest (scikit-learn)
#   1) (Optional) Install deps if needed:
#        !pip -q install pandas numpy scikit-learn matplotlib torch \
#            --extra-index-url https://download.pytorch.org/whl/cpu
#   2) Upload and run:
#        %run insider_threat_pipeline.py
#   3) When prompted, upload either:
#        - One ZIP that contains CERT/, LANL/, ENRON/ with CSVs inside, OR
#        - Individual CSVs: logon.csv, file.csv, device.csv, auth.csv, emails.csv
# -----------------------------------------------------------------------------

import os, zipfile, shutil, random, re, gzip, bz2, tarfile
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Try to import torch (if not available, instruct user to install)
try:
    import torch
    import torch.nn as nn
except Exception as e:
    print("[WARN] PyTorch not available. If running in Colab, install with:")
    print("       !pip -q install torch --extra-index-url https://download.pytorch.org/whl/cpu")
    raise

# ------------------ Reproducibility & Dirs ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = Path("/content/data")
CERT_DIR, LANL_DIR, ENRON_DIR = BASE/"CERT", BASE/"LANL", BASE/"ENRON"
for p in [CERT_DIR, LANL_DIR, ENRON_DIR]:
    p.mkdir(parents=True, exist_ok=True)

ROW_LIMIT_CERT = None
ROW_LIMIT_LANL = None
ROW_LIMIT_ENRON = None
VAL_PCTL = 95  # percentile for AE anomaly threshold on validation set

# ------------------ Upload (Colab) ------------------
def do_upload():
    """Upload a ZIP (with CERT/, LANL/, ENRON/) or individual CSVs into /content/data."""
    try:
        from google.colab import files
    except Exception:
        print("[INFO] google.colab not detected. Skipping upload; expecting files on disk under /content/data.")
        return
    print("Upload a single ZIP (with CERT/, LANL/, ENRON/), or individual CSVs")
    uploaded = files.upload()
    for name, data in uploaded.items():
        p = BASE/name
        with open(p, "wb") as f: f.write(data)
        if zipfile.is_zipfile(p):
            print("Extracting:", name)
            with zipfile.ZipFile(p) as z: z.extractall(BASE)
    # Move loose CSVs if needed
    for name in uploaded.keys():
        src = BASE/name
        lname = name.lower()
        if lname.endswith("logon.csv"):  shutil.move(str(src), str(CERT_DIR/"logon.csv"))
        elif lname.endswith("file.csv"): shutil.move(str(src), str(CERT_DIR/"file.csv"))
        elif lname.endswith("device.csv"): shutil.move(str(src), str(CERT_DIR/"device.csv"))
        elif lname.endswith("auth.csv"): shutil.move(str(src), str(LANL_DIR/"auth.csv"))
        elif lname.endswith("emails.csv"): shutil.move(str(src), str(ENRON_DIR/"emails.csv"))
    print("CERT files:", list(CERT_DIR.glob("*.csv")))
    print("LANL files:", list(LANL_DIR.glob("*.csv")))
    print("ENRON files:", list(ENRON_DIR.glob("*.csv")))

# ------------------ Load ------------------
def read_csv(path, parse_dates=None, nrows=None, usecols=None):
    p = Path(path)
    if not p.exists():
        print("Missing:", p)
        return None
    return pd.read_csv(p, low_memory=False, parse_dates=parse_dates, nrows=nrows, usecols=usecols)

# ------------------ Sessionization ------------------
def sessionize_cert(logon, files_, device):
    if any(df is None for df in [logon, files_, device]): return None
    def rn(df, m):
        m = {k:v for k,v in m.items() if k in df.columns and v not in df.columns}
        return df.rename(columns=m) if m else df
    logon  = rn(logon,  {'event':'activity','computer':'pc','id':'user','date':'time'})
    files_ = rn(files_, {'activity':'operation','date':'time'})
    device = rn(device, {'activity':'action','date':'time'})
    for df in (logon, files_, device):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    logon['week']  = logon['time'].dt.to_period('W').dt.start_time
    files_['week'] = files_['time'].dt.to_period('W').dt.start_time
    device['week'] = device['time'].dt.to_period('W').dt.start_time
    g1 = logon.groupby(['user','week'], sort=False).agg(
        logons=('activity', lambda s: (s.astype(str).str.lower().eq('logon')).sum()),
        logoffs=('activity', lambda s: (s.astype(str).str.lower().eq('logoff')).sum()),
        unique_pcs=('pc','nunique'),
        offhour_logons=('time', lambda c: ((c.dt.hour<8)|(c.dt.hour>19)).sum())
    ).reset_index()
    g2 = files_.groupby(['user','week'], sort=False).agg(
        file_ops=('operation','count'),
        file_writes=('operation', lambda s: (s.astype(str).str.lower().eq('write')).sum()),
        bytes_total=('bytes','sum') if 'bytes' in files_.columns else ('operation','count')
    ).reset_index()
    g3 = device.groupby(['user','week'], sort=False).agg(
        usb_events=('action','count'),
        usb_mounts=('action', lambda s: (s.astype(str).str.lower().eq('mount')).sum())
    ).reset_index()
    return g1.merge(g2, on=['user','week'], how='left').merge(g3, on=['user','week'], how='left').fillna(0)

def sessionize_lanl(auth):
    if auth is None: return None
    a = auth.copy()
    for src in ['user','user_name','source_user','account']:
        if src in a.columns: a.rename(columns={src:'user'}, inplace=True); break
    for sh in ['src_host','source_host','src','computer']:
        if sh in a.columns: a.rename(columns={sh:'src_host'}, inplace=True); break
    for dh in ['dst_host','dest_host','destination_host','target_host']:
        if dh in a.columns: a.rename(columns={dh:'dst_host'}, inplace=True); break
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
    return e.groupby(['user','week'], sort=False).agg(
        emails=('subject','count'),
        mean_len=('len_body','mean'),
        kw_hits=('kw_hits','sum')
    ).reset_index()

# ------------------ Scale & Split ------------------
def to_X(df, cols):
    X = df[cols].astype('float32').values
    scaler = StandardScaler().fit(X)
    return scaler.transform(X).astype('float32'), scaler

def split_idx(n, train=0.8, val=0.1):
    idx = np.arange(n); np.random.shuffle(idx)
    ntr = int(n*train); nva = int(n*val)
    return idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

# ------------------ Autoencoder ------------------
class AE(nn.Module):
    def __init__(self, d, h=None, bottleneck=8):
        super().__init__()
        h = h or max(16, 2*d)
        self.enc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck,h), nn.ReLU(), nn.Linear(h,d))
    def forward(self, x): return self.dec(self.enc(x))

def train_ae(Xtr, Xva, d, epochs=15, lr=1e-3, bs=256):
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

# ------------------ Isolation Forest ------------------
def iso_score(Xtr, Xte):
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iso.fit(Xtr)
    s = -iso.decision_function(Xte)
    # NumPy 2.0 safe normalization
    return (s - s.min()) / (np.ptp(s) + 1e-9)

# ------------------ Visuals ------------------
def plot_hist(values, title, xlabel, thr=None):
    plt.figure()
    plt.hist(values, bins=50)
    if thr is not None:
        ymin, ymax = plt.ylim()
        plt.vlines(thr, ymin, ymax)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.show()

def plot_scatter(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.6)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.show()

# ------------------ Main ------------------
def main():
    do_upload()

    # Load CSVs
    logon  = read_csv(CERT_DIR/"logon.csv",  parse_dates=["time"], nrows=ROW_LIMIT_CERT)
    files_ = read_csv(CERT_DIR/"file.csv",   parse_dates=["time"], nrows=ROW_LIMIT_CERT)
    device = read_csv(CERT_DIR/"device.csv", parse_dates=["time"], nrows=ROW_LIMIT_CERT)
    auth   = read_csv(LANL_DIR/"auth.csv",   parse_dates=["time"], nrows=ROW_LIMIT_LANL)
    emails = read_csv(ENRON_DIR/"emails.csv",parse_dates=["date"], nrows=ROW_LIMIT_ENRON)

    print("Shapes — CERT:", *(x.shape if x is not None else None for x in [logon, files_, device]),
          "| LANL:", None if auth is None else auth.shape, "| ENRON:", None if emails is None else emails.shape)

    # Sessionize
    cert_s = sessionize_cert(logon, files_, device)
    lanl_s = sessionize_lanl(auth)
    enron_s= sessionize_enron(emails)

    print("Sessionized — CERT:", None if cert_s is None else cert_s.shape,
          "LANL:", None if lanl_s is None else lanl_s.shape,
          "ENRON:", None if enron_s is None else enron_s.shape)

    # Scale & Split (keep test keys for alignment)
    cert_cols  = ['logons','logoffs','unique_pcs','offhour_logons','file_ops','file_writes','bytes_total','usb_events','usb_mounts']
    lanl_cols  = ['auth_events','auth_success','uniq_src','uniq_dst','night_auth','success_rate']
    enron_cols = ['emails','mean_len','kw_hits']

    blocks, blocks_meta = {}, {}

    if cert_s is not None and set(cert_cols).issubset(cert_s.columns):
        Xc, _ = to_X(cert_s, cert_cols); tr,va,te = split_idx(len(Xc))
        blocks['CERT'] = (Xc[tr], Xc[va], Xc[te])
        ck = cert_s[['user','week']].iloc[te].copy()
        ck['key'] = ck['user'].astype(str) + '|' + ck['week'].astype(str)
        blocks_meta['CERT'] = {'te_keys': ck['key'].to_numpy()}

    if lanl_s is not None and set(lanl_cols).issubset(lanl_s.columns):
        Xl, _ = to_X(lanl_s, lanl_cols); tr,va,te = split_idx(len(Xl))
        blocks['LANL'] = (Xl[tr], Xl[va], Xl[te])
        lk = lanl_s[['user','week']].iloc[te].copy()
        lk['key'] = lk['user'].astype(str) + '|' + lk['week'].astype(str)
        blocks_meta['LANL'] = {'te_keys': lk['key'].to_numpy()}

    if enron_s is not None and set(enron_cols).issubset(enron_s.columns):
        Xe, _ = to_X(enron_s, enron_cols); tr,va,te = split_idx(len(Xe))
        blocks['ENRON'] = (Xe[tr], Xe[va], Xe[te])
        ek = enron_s[['user','week']].iloc[te].copy()
        ek['key'] = ek['user'].astype(str) + '|' + ek['week'].astype(str)
        blocks_meta['ENRON'] = {'te_keys': ek['key'].to_numpy()}

    print("Prepared blocks:", {k: tuple(x.shape for x in v) for k,v in blocks.items()})

    # Autoencoder training
    scores = {}
    for name, (Xtr, Xva, Xte) in blocks.items():
        print(f"[{name}] training AE d={Xtr.shape[1]}  ntr={len(Xtr)} nva={len(Xva)}")
        m, dev = train_ae(Xtr, Xva, Xtr.shape[1])
        e_va = recon_err(m, Xva, dev)
        thr  = float(np.percentile(e_va, VAL_PCTL))
        e_te = recon_err(m, Xte, dev)
        flags = (e_te > thr).astype(int)
        scores[name] = dict(thr=thr, err_val=e_va, err_test=e_te, flags=flags)

    print({k: (v['err_test'].shape, v['thr']) for k,v in scores.items()})

    # Isolation Forest + aligned fusion
    for name, (Xtr, Xva, Xte) in blocks.items():
        scores[name]['iso'] = iso_score(Xtr, Xte)

    # Build per-dataset frames with keys for alignment
    per_ds = []
    for name in scores:
        if name not in blocks_meta or 'te_keys' not in blocks_meta[name]:
            print(f"[WARN] Missing keys for {name}; skipping in fusion.")
            continue
        k = blocks_meta[name]['te_keys']
        df = pd.DataFrame({
            'key': k,
            f'{name}_ae_flag': scores[name]['flags'].astype(int),
            f'{name}_iso': scores[name]['iso'].astype(float),
            f'{name}_ae_err': scores[name]['err_test'].astype(float)
        })
        per_ds.append(df)

    if per_ds:
        fused_df = per_ds[0]
        for df in per_ds[1:]:
            fused_df = fused_df.merge(df, on='key', how='outer')

        ae_cols  = [c for c in fused_df.columns if c.endswith('_ae_flag')]
        iso_cols = [c for c in fused_df.columns if c.endswith('_iso')]

        fused_df['models_present'] = fused_df[ae_cols].notna().sum(axis=1)
        fused_df['votes'] = fused_df[ae_cols].fillna(0).sum(axis=1)
        fused_df['fused_flag'] = (fused_df['votes'] >= np.ceil(fused_df['models_present']/2)).astype(int)

        # Composite anomaly score (mean of signals)
        sig_cols = [c for c in fused_df.columns if c.endswith('_ae_err')] + iso_cols
        if sig_cols:
            fused_df['composite_score'] = fused_df[sig_cols].mean(axis=1, skipna=True)

        print("Fused anomaly rate:", float(fused_df['fused_flag'].mean()))
    else:
        fused_df = None
        print("No datasets available for fusion; skipping.")

    # Visualizations
    for name in scores:
        sc = scores[name]
        plot_hist(sc['err_test'], f"{name} — AE reconstruction error (test)", "Reconstruction error", thr=sc.get('thr'))
        if 'iso' in sc:
            plot_hist(sc['iso'], f"{name} — IsolationForest anomaly score (test)", "Normalized ISO score")
            plot_scatter(sc['err_test'], sc['iso'], f"{name} — AE error vs ISO score", "AE reconstruction error", "ISO anomaly score")

    if fused_df is not None and len(fused_df):
        cols_show = ['key', 'fused_flag'] + \
                    [c for c in fused_df.columns if c.endswith('_ae_err')] + \
                    [c for c in fused_df.columns if c.endswith('_iso')] + \
                    (['composite_score'] if 'composite_score' in fused_df.columns else [])
        topN = fused_df.sort_values(by=['fused_flag','composite_score'], ascending=[False, False]).head(20)[cols_show]
        try:
            from IPython.display import display
            print("\nTop-20 anomalies (aligned across datasets):")
            display(topN)
        except Exception:
            print("\nTop-20 anomalies (aligned across datasets):")
            print(topN.to_string(index=False))
    else:
        print("\n[INFO] Fusion not available; skipping Top‑N table.")

if __name__ == "__main__":
    main()
