# insider_threat_pipeline_mitre.py
# -----------------------------------------------------------------------------
# Colab-ready upload-based pipeline for Insider Threat anomaly detection
# Datasets: CERT (logon/file/device), LANL (auth), ENRON (emails)
# Models: Autoencoder (PyTorch) + Isolation Forest (scikit-learn)
# Steps: Upload -> Load -> Sessionize -> Scale/Split -> Train AE -> ISO ->
#        Aligned Fusion -> Visualizations -> MITRE ATT&CK Mapping
# -----------------------------------------------------------------------------

import os, zipfile, shutil, random
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Try to import torch; fail fast with a helpful message if missing
try:
    import torch
    import torch.nn as nn
except Exception:
    raise ImportError("PyTorch is required. In Colab run: !pip -q install torch --extra-index-url https://download.pytorch.org/whl/cpu")

# ------------------ Reproducibility & Dirs ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE = Path("/content/data")
CERT_DIR, LANL_DIR, ENRON_DIR = BASE/"CERT", BASE/"LANL", BASE/"ENRON"
for p in [CERT_DIR, LANL_DIR, ENRON_DIR]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/content/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROW_LIMIT_CERT = None   # set small ints for quick tests (e.g., 50_000)
ROW_LIMIT_LANL = None
ROW_LIMIT_ENRON = None
VAL_PCTL = 95  # percentile for AE anomaly threshold on validation set

# ------------------ Upload (Colab) ------------------
def do_upload():
    """Upload a ZIP (with CERT/, LANL/, ENRON/) or individual CSVs into /content/data."""
    try:
        from google.colab import files
    except Exception:
        print("[INFO] google.colab not detected. Skipping upload; expecting files already under /content/data.")
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
        if lname.endswith("logon.csv"):   shutil.move(str(src), str(CERT_DIR/"logon.csv"))
        elif lname.endswith("file.csv"):  shutil.move(str(src), str(CERT_DIR/"file.csv"))
        elif lname.endswith("device.csv"):shutil.move(str(src), str(CERT_DIR/"device.csv"))
        elif lname.endswith("auth.csv"):  shutil.move(str(src), str(LANL_DIR/"auth.csv"))
        elif lname.endswith("emails.csv"):shutil.move(str(src), str(ENRON_DIR/"emails.csv"))
    print("CERT files:", [p.name for p in CERT_DIR.glob('*.csv')])
    print("LANL files:", [p.name for p in LANL_DIR.glob('*.csv')])
    print("ENRON files:", [p.name for p in ENRON_DIR.glob('*.csv')])

# ------------------ Load ------------------
def read_csv(path, parse_dates=None, nrows=None, usecols=None):
    p = Path(path)
    if not p.exists():
        print("Missing:", p)
        return None
    return pd.read_csv(p, low_memory=False, parse_dates=parse_dates, nrows=nrows, usecols=usecols)

# ------------------ Sessionization ------------------
def sessionize_cert(logon, files_, device):
    if any(df is None for df in [logon, files_, device]): 
        print("[CERT] Missing one or more source CSVs; skipping CERT sessionization.")
        return None

    # Normalization of column names if datasets differ
    def rn(df, m):
        m = {k:v for k,v in m.items() if k in df.columns and v not in df.columns}
        return df.rename(columns=m) if m else df

    logon  = rn(logon,  {'event':'activity','computer':'pc','id':'user','date':'time'})
    files_ = rn(files_, {'activity':'operation','date':'time'})
    device = rn(device, {'activity':'action','date':'time'})

    for df in (logon, files_, device):
        if 'time' not in df.columns:
            raise ValueError("Expected a 'time' column in CERT CSVs. Add/rename appropriately.")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Weekly aggregation
    for df in (logon, files_, device):
        df['week'] = df['time'].dt.to_period('W').dt.start_time

    g1 = logon.groupby(['user','week'], sort=False).agg(
        logons=('activity', lambda s: (s.astype(str).str.lower().eq('logon')).sum() if 'activity' in logon.columns else len(s)),
        logoffs=('activity', lambda s: (s.astype(str).str.lower().eq('logoff')).sum() if 'activity' in logon.columns else 0),
        unique_pcs=('pc','nunique') if 'pc' in logon.columns else ('user','count'),
        offhour_logons=('time', lambda c: ((c.dt.hour<8)|(c.dt.hour>19)).sum())
    ).reset_index()

    g2 = files_.groupby(['user','week'], sort=False).agg(
        file_ops=('operation','count') if 'operation' in files_.columns else ('user','count'),
        file_writes=('operation', lambda s: (s.astype(str).str.lower().eq('write')).sum() if 'operation' in files_.columns else 0),
        bytes_total=('bytes','sum') if 'bytes' in files_.columns else ('operation','count')
    ).reset_index()

    g3 = device.groupby(['user','week'], sort=False).agg(
        usb_events=('action','count') if 'action' in device.columns else ('user','count'),
        usb_mounts=('action', lambda s: (s.astype(str).str.lower().eq('mount')).sum() if 'action' in device.columns else 0)
    ).reset_index()

    out = g1.merge(g2, on=['user','week'], how='left').merge(g3, on=['user','week'], how='left').fillna(0)
    return out

def sessionize_lanl(auth):
    if auth is None:
        print("[LANL] Missing auth.csv; skipping LANL sessionization.")
        return None
    a = auth.copy()

    # Normalize column names
    if 'user' not in a.columns:
        for src in ['user_name','source_user','account']:
            if src in a.columns: a.rename(columns={src:'user'}, inplace=True); break
    if 'src_host' not in a.columns:
        for sh in ['source_host','src','computer']: 
            if sh in a.columns: a.rename(columns={sh:'src_host'}, inplace=True); break
    if 'dst_host' not in a.columns:
        for dh in ['dest_host','destination_host','target_host']: 
            if dh in a.columns: a.rename(columns={dh:'dst_host'}, inplace=True); break
    if 'success' not in a.columns and 'status' in a.columns:
        a['success'] = a['status'].astype(str).str.lower().isin(['success','ok','true','1']).astype(int)

    a['time'] = pd.to_datetime(a['time'], errors='coerce')
    a['week'] = a['time'].dt.to_period('W').dt.start_time

    g = a.groupby(['user','week'], sort=False).agg(
        auth_events=('success','count') if 'success' in a.columns else ('user','count'),
        auth_success=('success','sum') if 'success' in a.columns else ('user','count'),
        uniq_src=('src_host','nunique') if 'src_host' in a.columns else ('user','count'),
        uniq_dst=('dst_host','nunique') if 'dst_host' in a.columns else ('user','count'),
        night_auth=('time', lambda c: ((c.dt.hour<7)|(c.dt.hour>20)).sum())
    ).reset_index()

    g['success_rate'] = (g['auth_success'] / g['auth_events']).replace([np.inf, -np.inf], 0).fillna(0)
    return g

def sessionize_enron(emails):
    if emails is None:
        print("[ENRON] Missing emails.csv; skipping ENRON sessionization.")
        return None
    e = emails.copy()

    # Normalize column names
    if 'user' not in e.columns:
        if 'from' in e.columns:
            e['user'] = e['from'].astype(str).str.extract(r'(^[^@\s>]+)')
        elif 'sender' in e.columns:
            e['user'] = e['sender'].astype(str).str.extract(r'(^[^@\s>]+)')
    if 'date' not in e.columns:
        for c in ['Date','datetime','time']:
            if c in e.columns: e.rename(columns={c:'date'}, inplace=True); break

    e['date'] = pd.to_datetime(e['date'], errors='coerce')
    e['week'] = e['date'].dt.to_period('W').dt.start_time

    if 'body' in e.columns:
        e['len_body'] = e['body'].astype(str).str.len()
        kw = r'\b(resign|quit|leaving|angry|frustrat|complain)\b'
        e['kw_hits'] = e['body'].astype(str).str.lower().str.contains(kw, regex=True).astype(int)
    else:
        e['len_body'] = 0
        e['kw_hits'] = 0

    out = e.groupby(['user','week'], sort=False).agg(
        emails=('subject','count') if 'subject' in e.columns else ('user','count'),
        mean_len=('len_body','mean'),
        kw_hits=('kw_hits','sum')
    ).reset_index()
    return out

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
        if va_loss < best[0]:
            best = (va_loss, model.state_dict())
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

# ------------------ MITRE ATT&CK Mapping ------------------
def mitre_mapping(fused_df, cert_s, lanl_s, enron_s, blocks_meta):
    if fused_df is None or fused_df.empty:
        print("[MITRE] fused_df not available; skipping MITRE mapping.")
        return None, None, None

    def slice_test(df, te_keys):
        df_ = df.copy()
        df_['key'] = df_['user'].astype(str) + '|' + df_['week'].astype(str)
        return df_.loc[df_['key'].isin(te_keys)].drop_duplicates('key')

    mitre_frames = []
    if 'CERT' in blocks_meta and cert_s is not None:
        cert_te_keys = set(blocks_meta['CERT']['te_keys'])
        cert_view = slice_test(cert_s[['user','week','logons','logoffs','unique_pcs',
                                       'offhour_logons','file_ops','file_writes',
                                       'bytes_total','usb_events','usb_mounts']],
                               cert_te_keys)
        mitre_frames.append(cert_view)

    if 'LANL' in blocks_meta and lanl_s is not None:
        lanl_te_keys = set(blocks_meta['LANL']['te_keys'])
        lanl_view = slice_test(lanl_s[['user','week','auth_events','auth_success',
                                       'success_rate','uniq_src','uniq_dst','night_auth']],
                               lanl_te_keys)
        mitre_frames.append(lanl_view)

    if 'ENRON' in blocks_meta and enron_s is not None:
        enron_te_keys = set(blocks_meta['ENRON']['te_keys'])
        enron_view = slice_test(enron_s[['user','week','emails','mean_len','kw_hits']],
                                enron_te_keys)
        mitre_frames.append(enron_view)

    if mitre_frames:
        mitre_base = mitre_frames[0]
        for extra in mitre_frames[1:]:
            mitre_base = mitre_base.merge(
                extra[['key'] + [c for c in extra.columns if c not in ['user','week']]],
                on='key', how='outer'
            )
    else:
        mitre_base = pd.DataFrame(columns=['key'])

    def nz(x): return 0 if pd.isna(x) else x

    def indicators(row):
        ind = {}
        # CERT indicators
        ind['offhours_access']   = nz(row.get('offhour_logons',0)) >= 3
        ind['multi_host_access'] = nz(row.get('unique_pcs',0))     >= 4
        ind['write_burst']       = nz(row.get('file_writes',0))    >= 40 or nz(row.get('file_ops',0)) >= 100
        ind['large_bytes']       = nz(row.get('bytes_total',0))    >= 5e8
        ind['usb_activity']      = nz(row.get('usb_mounts',0))     >= 1
        # LANL indicators
        ind['auth_fail_spike']   = (nz(row.get('auth_events',1)) - nz(row.get('auth_success',0))) >= 6
        ind['low_success_rate']  = nz(row.get('success_rate',1.0)) < 0.75
        ind['night_auth_spike']  = nz(row.get('night_auth',0))     >= 5
        ind['host_sprawl']       = (nz(row.get('uniq_src',0)) >= 5) or (nz(row.get('uniq_dst',0)) >= 7)
        # ENRON indicators
        ind['neg_keyword_spike'] = nz(row.get('kw_hits',0))        >= 5
        ind['email_surge']       = nz(row.get('emails',0))         >= 80
        return ind

    ind_df = mitre_base.apply(indicators, axis=1, result_type='expand')
    mitre_feat = pd.concat([mitre_base[['key']], ind_df], axis=1)

    MITRE_CATALOG = {
        'offhours_access':   {'tactic': 'Credential Access', 'technique': 'Valid Accounts', 'tech_id': 'T1078'},
        'auth_fail_spike':   {'tactic': 'Credential Access', 'technique': 'Brute Force', 'tech_id': 'T1110'},
        'low_success_rate':  {'tactic': 'Credential Access', 'technique': 'Password Spraying/Guessing', 'tech_id': 'T1110'},
        'multi_host_access': {'tactic': 'Lateral Movement',  'technique': 'Remote Services / Lateral Movement', 'tech_id': 'T1021'},
        'host_sprawl':       {'tactic': 'Lateral Movement',  'technique': 'Internal Pivoting / Discovery',      'tech_id': 'T1046'},
        'write_burst':       {'tactic': 'Collection',        'technique': 'Data from Local System',            'tech_id': 'T1005'},
        'large_bytes':       {'tactic': 'Exfiltration',      'technique': 'Exfiltration Over Network',         'tech_id': 'T1041'},
        'usb_activity':      {'tactic': 'Exfiltration',      'technique': 'Exfiltration via Removable Media',  'tech_id': 'T1052'},
        'night_auth_spike':  {'tactic': 'Defense Evasion',   'technique': 'Use of Unusual Hours',              'tech_id': 'T1036'},
        'neg_keyword_spike': {'tactic': 'Collection',        'technique': 'Sensitive Data Signals in Email',   'tech_id': '—'},
        'email_surge':       {'tactic': 'Collection',        'technique': 'Bulk Emailing (Staging)',           'tech_id': '—'},
    }

    def map_to_mitre(row):
        tags = []
        for ind_name, fired in row.items():
            if ind_name == 'key': 
                continue
            if bool(fired) and ind_name in MITRE_CATALOG:
                info = MITRE_CATALOG[ind_name]
                tags.append(f"{info['tactic']} | {info['technique']} ({info['tech_id']})")
        return sorted(set(tags))

    mitre_feat['mitre_tags'] = ind_df.apply(map_to_mitre, axis=1)

    return mitre_feat, ind_df, MITRE_CATALOG

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
          "LANL:", None if lanl_s is not None and hasattr(lanl_s,'shape') else None,
          "ENRON:", None if enron_s is None else enron_s.shape)

    # Feature columns
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

    # MITRE mapping
    mitre_feat, ind_df, MITRE_CATALOG = mitre_mapping(fused_df, cert_s, lanl_s, enron_s, blocks_meta) if fused_df is not None else (None,None,None)
    if mitre_feat is not None:
        mitre_join = fused_df.merge(mitre_feat[['key'] + [c for c in mitre_feat.columns if c!='key']], on='key', how='left')

        if 'composite_score' in mitre_join.columns:
            rank_cols = ['fused_flag','composite_score']
        else:
            rank_cols = ['fused_flag']

        top_mitre = mitre_join.sort_values(by=rank_cols, ascending=[False] + [False]*(len(rank_cols)-1)) \
                              .head(30) \
                              [['key','fused_flag','mitre_tags'] + [c for c in mitre_join.columns if c.endswith('_ae_err') or c.endswith('_iso')]]

        print("\n[MITRE] Top-30 fused anomalies with ATT&CK mappings:")
        try:
            from IPython.display import display
            display(top_mitre)
        except Exception:
            print(top_mitre.to_string(index=False))

        # Frequency of tactics/techniques among anomalies only
        def explode_tags(series):
            all_tags = []
            for lst in series.fillna([]):
                all_tags.extend(lst if isinstance(lst, list) else [])
            return all_tags

        anomalous = mitre_join.loc[mitre_join['fused_flag']==1, 'mitre_tags']
        tag_counts = pd.Series(explode_tags(anomalous)).value_counts().reset_index()
        tag_counts.columns = ['ATT&CK tactic | technique (ID)', 'count']

        print("\n[MITRE] ATT&CK mapping frequency among fused anomalies:")
        try:
            from IPython.display import display
            display(tag_counts.head(20))
        except Exception:
            print(tag_counts.head(20).to_string(index=False))

        # Save outputs
        fused_path = OUT_DIR/"fused_decisions.csv"
        mitre_path = OUT_DIR/"mitre_mapping.csv"
        fused_df.to_csv(fused_path, index=False)
        mitre_join.to_csv(mitre_path, index=False)
        print(f"\nSaved outputs:\n  {fused_path}\n  {mitre_path}")
    else:
        print("[MITRE] Skipped mapping (no fused_df).")

if __name__ == "__main__":
    main()
