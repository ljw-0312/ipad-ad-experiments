# Standalone sparse-endpoint V-JEPA 2 CPU exploratory comparison.
import hashlib, json, csv, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
def fit_space(rows,rank,standardize=False):
    x=np.asarray(rows,dtype=np.float64)
    if x.ndim!=2 or len(x)<8 or not np.isfinite(x).all():
        raise ValueError('Need >=8 finite normal observations per bank')
    center=x.mean(0)
    scale=np.ones(x.shape[1])
    if standardize:
        # Temporal descriptor is 10*4 positions/sizes + 9*2 velocities.
        floor=np.r_[np.full(40,.01),np.full(18,.001)]
        if x.shape[1]!=58:raise ValueError('Expected 10-frame, 58-dimensional motion descriptor')
        scale=np.maximum(x.std(0),floor)
    y=(x-center)/scale
    _,s,vt=np.linalg.svd(y,full_matrices=False)
    numerical=int(np.sum(s>max(float(s[0]),1)*1e-10))
    r=min(rank,len(x)-2,x.shape[1]-1,numerical)
    return dict(center=center.tolist(),scale=scale.tolist(),basis=vt[:r].tolist(),rank=r,n=len(x))

def residual(row,bank):
    x=(np.asarray(row,dtype=float)-bank['center'])/bank['scale']
    if not np.isfinite(x).all():raise ValueError('Nonfinite feature')
    if bank['rank']:
        u=np.asarray(bank['basis'])
        x=x-(x@u.T)@u
    return float(np.linalg.norm(x)/np.sqrt(x.size))

def rank_metrics(labels,scores):
    """AUROC with average ties; average precision grouped at distinct scores."""
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float)
    if len(y)!=len(s) or not np.isfinite(s).all() or not set(y)<={0,1}:
        raise ValueError('Invalid labels/scores')
    positives=int(y.sum());negatives=len(y)-positives
    if not positives or not negatives:return dict(auroc=None,average_precision=None)
    order=np.argsort(s,kind='stable');ranks=np.empty(len(s),dtype=float)
    start=0
    while start<len(s):
        end=start+1
        while end<len(s) and s[order[end]]==s[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2
        start=end
    auc=(ranks[y==1].sum()-positives*(positives+1)/2)/(positives*negatives)
    order=np.argsort(-s,kind='stable');tp=seen=0;ap=0.;start=0
    while start<len(s):
        end=start+1
        while end<len(s) and s[order[end]]==s[order[start]]:end+=1
        added=int(y[order[start:end]].sum());tp+=added;seen=end
        ap+=(added/positives)*(tp/seen)
        start=end
    return dict(auroc=float(auc),average_precision=float(ap))

def decision_metrics(labels,scores,threshold,all_labels=None):
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float);p=s>threshold
    tp=int(np.sum((y==1)&p));fp=int(np.sum((y==0)&p));fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p))
    total_pos=int(np.sum(all_labels)) if all_labels is not None else int(y.sum())
    return dict(tp=tp,fp=fp,fn_known=fn,tn=tn,precision=tp/max(1,tp+fp),
        recall_known=tp/max(1,tp+fn),recall_all=tp/max(1,total_pos),
        normal_frame_false_positive_rate_known=fp/max(1,fp+tn))

def nearest_distance(query, bank, chunk=128):
    q=np.asarray(query,dtype=np.float32);b=np.asarray(bank,dtype=np.float32)
    if q.ndim!=2 or b.ndim!=2 or q.shape[1]!=b.shape[1] or not len(b):
        raise ValueError('Invalid nearest-normal arrays')
    if not np.isfinite(q).all() or not np.isfinite(b).all():raise ValueError('Nonfinite arrays')
    result=[];b2=np.sum(b*b,axis=1)
    for start in range(0,len(q),chunk):
        x=q[start:start+chunk]
        d=np.sum(x*x,axis=1)[:,None]+b2[None,:]-2*x@b.T
        result.extend(np.sqrt(np.maximum(d.min(axis=1),0)/b.shape[1]).tolist())
    return result

def load_cache(project,settings):
    expected=settings['source_signature'];records={}
    paths=list((project/'meeting_experiments/feature_cache').glob('*.npz'))
    for i,path in enumerate(paths,1):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()+meta['id'].encode()).hexdigest()
            if path.stem!=key or meta['signature']!=key:continue
            identity=(meta['split'],meta['sequence'])
            if identity in records:raise ValueError('Duplicate cache identity')
            for f,z in zip(meta['frames'],data['global']):f['global']=z.tolist()
            for f in meta['frames']:
                for o in f['objects']:o['appearance']=data['objects'][o.pop('feature_row')].tolist()
            records[identity]=meta
        if i%10==0:print(f'Cache checked {i}/{len(paths)}',flush=True)
    normal=[records.get(('training',f'{i:02}')) for i in range(1,35)]
    tested=[records.get(('testing',f'{i:02}')) for i in range(1,16)]
    if any(r is None for r in normal+tested):raise RuntimeError('Matching caches incomplete. This script will not rerun detection.')
    ids=[r['id'] for r in normal+tested]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate train/test video content')
    return normal,tested

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def revision_from_info(info):
    revision = getattr(info, 'sha', None)
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdefABCDEF' for c in revision):
        raise RuntimeError('Hugging Face did not return a valid model commit. Check the connection to huggingface.co.')
    return revision

def restore_frames(project, settings):
    """Restore only the source R01 JPGs and labels; never run detection."""
    root = Path('/content/meeting_ad_data') / settings['data_manifest'][:16] / 'R01'
    archive = project / 'IPAD_dataset.zip'
    print('Checking original dataset ZIP and local R01 files...', flush=True)
    with zipfile.ZipFile(archive) as z:
        selected = []
        for item in z.infolist():
            if item.is_dir():
                continue
            parts = Path(item.filename.replace(chr(92), '/')).parts
            if 'R01' not in parts or '__MACOSX' in parts:
                continue
            rel = parts[parts.index('R01'):]
            frame = len(rel) == 5 and rel[1] in ['training', 'testing'] and rel[2] == 'frames' and rel[4].endswith('.jpg')
            label = len(rel) == 3 and rel[1] == 'test_label' and rel[2].endswith('.npy')
            if frame or label:
                selected.append((Path(*rel).as_posix(), item))
        if len(selected) != len({name for name, _ in selected}):
            raise ValueError('Duplicate dataset entries')
        key = hashlib.sha256(json.dumps([(n, i.CRC, i.file_size) for n, i in sorted(selected)]).encode()).hexdigest()
        if key != settings['data_manifest']:
            raise ValueError('Dataset ZIP differs from the original experiment')
        restored = 0
        for name, item in selected:
            target = root.parent / name
            if not target.resolve().is_relative_to(root.parent.resolve()):
                raise ValueError('Unsafe dataset path')
            if not target.is_file() or target.stat().st_size != item.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(item))
                restored += 1
                if restored % 1000 == 0:
                    print(f'Restored {restored} R01 files...', flush=True)
    print(f'R01 local files ready; restored {restored} files. Detector is not run.', flush=True)
    return root

def source_paths(record, root):
    folder = root / record['split'] / 'frames' / record['sequence']
    paths = sorted(folder.glob('*.jpg'), key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(len(record['frames']))):
        raise ValueError(f'Frame alignment mismatch: {folder}')
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if digest.hexdigest() != record['id']:
        raise ValueError(f'JPG content differs from cached detector source: {folder}')
    return paths
import gc
import os
import sys
import subprocess
import time

VJEPA_MODEL = 'facebook/vjepa2-vitl-fpc64-256'
SOURCE_FOLDER = 'R01_20261007_102640_886381_UTC'
DINO_FOLDER = 'dinov2_comparison_20261008_002809_792348_UTC'
WINDOW = int(os.environ.get('IPAD_VJEPA_WINDOW', '16'))
STEP = int(os.environ.get('IPAD_VJEPA_STEP', '32'))
SIZE = 256
METHODS = ['G', 'T', 'C', 'V', 'GV']


def prepare_vjepa_dependencies():
    # A fresh process uses a separate package directory; the notebook's CLIP/DINO environment is not replaced.
    target = Path('/content') / f'ipad_vjepa_deps_py{sys.version_info.major}{sys.version_info.minor}_4571'
    marker = target / '.complete'
    if not marker.is_file():
        print('Preparing isolated V-JEPA packages. Existing torch is retained.', flush=True)
        packages = ['transformers==4.57.1', 'huggingface-hub==0.36.0', 'tokenizers==0.22.1', 'safetensors==0.6.2']
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target', str(target),
            '--upgrade', '--no-deps', '--disable-pip-version-check', *packages])
        marker.write_text('transformers=4.57.1\n', encoding='utf-8')
    sys.path.insert(0, str(target))


def endpoint_indices(length, window=WINDOW, step=STEP):
    if window < 2 or window % 2 or step < 1:
        raise ValueError('Window must be positive/even and step >=1')
    return list(range(window-1, length, step))


def normalize_feature(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Invalid video feature')
    norm = float(np.linalg.norm(x))
    if norm < 1e-8:
        raise ValueError('Zero video feature')
    return x / norm


def dino_temporal(features):
    # A cheap temporal reference from cached image features; no extra neural inference.
    z = np.asarray(features, dtype=np.float32)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError('Need multiple consecutive DINO frames')
    blocks = [z.mean(0), z[-1]-z[0], np.abs(np.diff(z, axis=0)).mean(0)]
    blocks = [b/max(float(np.linalg.norm(b)), 1e-6) for b in blocks]
    return normalize_feature(np.concatenate(blocks))


def load_dino_global(record, cache, signature):
    key = hashlib.sha256((record['signature']+json.dumps(signature, sort_keys=True)).encode()).hexdigest()
    path = cache / (key+'.npz')
    if not path.is_file():
        raise FileNotFoundError(f'Missing existing DINO cache: {record["split"]}/{record["sequence"]}')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta['source_id'] != record['id'] or meta['signature'] != signature:
            raise ValueError('DINO cache provenance mismatch')
        z = data['global'].copy()
    if z.shape != (len(record['frames']), signature['feature_dim']) or not np.isfinite(z).all():
        raise ValueError('DINO frame/features mismatch')
    return z


def video_pixels(paths, size=SIZE):
    """Same physical center crop throughout a clip. Native V-JEPA normalization; matched crop geometry to DINO."""
    frames = []
    for path in paths:
        with Image.open(path) as im:
            pic = im.convert('RGB')
            factor = size/min(pic.size)
            w, h = round(pic.width*factor), round(pic.height*factor)
            pic = pic.resize((w, h), Image.Resampling.BICUBIC)
            left, top = (w-size)//2, (h-size)//2
            pic = pic.crop((left, top, left+size, top+size))
            frames.append(np.asarray(pic, dtype=np.float32)/255.)
    x = np.stack(frames).transpose(0, 3, 1, 2)
    mean = np.array([.485, .456, .406], np.float32)[None, :, None, None]
    std = np.array([.229, .224, .225], np.float32)[None, :, None, None]
    return (x-mean)/std


class VideoEncoder:
    def __init__(self):
        import torch
        from huggingface_hub import HfApi
        import transformers.utils.import_utils as optional
        # This runner consumes JPG arrays directly; unrelated audio/vision/generation extras are unnecessary.
        # Flags are specific to the pinned transformers 4.57.1 in this isolated process.
        optional._librosa_available = False
        optional._torchvision_available = False
        optional._sklearn_available = False
        from transformers import VJEPA2Config, VJEPA2Model
        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.revision = revision_from_info(HfApi().model_info(VJEPA_MODEL, revision='main'))
        print(f'Loading pretrained {VJEPA_MODEL} on CPU (~1.3 GB weight download on first use).', flush=True)
        config = VJEPA2Config.from_pretrained(VJEPA_MODEL, revision=self.revision)
        if config.crop_size != SIZE:
            raise ValueError('Input resolution must match the pretrained RoPE grid')
        self.model = VJEPA2Model.from_pretrained(VJEPA_MODEL, revision=self.revision, use_safetensors=True,
            attn_implementation='sdpa', dtype=torch.float32).to('cpu').eval()
        self.model.predictor = None  # get_vision_features skips it; prediction error is not used as anomaly score.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = config.hidden_size
        self.signature = dict(model=VJEPA_MODEL, revision=self.revision, transformers='4.57.1',
            window=WINDOW, endpoint_step=STEP, size=SIZE, feature_dim=self.dim, input='full-frame fixed center crop',
            pooling='mean over final encoder spatiotemporal tokens, L2 normalize', precision='float32',
            preprocessing='shortest side 256, center crop 256, bicubic, ImageNet normalize', version=1)
        gc.collect()

    def __call__(self, paths):
        torch = self.torch
        pixels = torch.from_numpy(video_pixels(paths)).unsqueeze(0)
        with torch.inference_mode():
            tokens = self.model.get_vision_features(pixels)
            if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != self.dim:
                raise ValueError('Unexpected V-JEPA encoder output shape')
            vector = tokens.mean(dim=1)[0].float().numpy()
        return normalize_feature(vector)


def extract_video_features(record, root, cache, encoder):
    paths = source_paths(record, root)
    indices = endpoint_indices(len(paths))
    key = hashlib.sha256((record['id']+json.dumps(encoder.signature, sort_keys=True)).encode()).hexdigest()
    target = cache/(key+'.npz');cache.mkdir(parents=True, exist_ok=True)
    features, timing = [], []
    if target.is_file():
        with np.load(target, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['signature'] != encoder.signature or meta['source_id'] != record['id']:
                raise ValueError('V-JEPA cache signature mismatch')
            saved = data['indices'].tolist()
            if saved != indices[:len(saved)] or data['features'].shape != (len(saved), encoder.dim):
                raise ValueError('V-JEPA partial cache alignment mismatch')
            features = data['features'].tolist()
            timing = meta['clip_seconds']
    resumed = len(features)
    for end in indices[resumed:]:
        tick = time.perf_counter()
        vector = encoder(paths[end-WINDOW+1:end+1])
        seconds = time.perf_counter()-tick
        features.append(vector.tolist());timing.append(seconds)
        meta = dict(signature=encoder.signature,source_id=record['id'],clip_seconds=timing,complete=len(features)==len(indices))
        temp = target.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, indices=np.array(indices[:len(features)]),features=np.asarray(features,np.float32),metadata=np.array(json.dumps(meta)))
        temp.replace(target)
        print(f"{record['split']}/{record['sequence']} clip {len(features)}/{len(indices)}: endpoint {end}, {seconds:.1f}s", flush=True)
    return np.asarray(features,np.float32),dict(sequence=record['sequence'],split=record['split'],clips=len(indices),
        resumed_clips=resumed,seconds=sum(timing),new_seconds=sum(timing[resumed:]),clip_seconds=timing)


def sampled_features(record, dino, video):
    endpoints = endpoint_indices(len(record['frames']))
    if len(video) != len(endpoints):
        raise ValueError('Video/endpoint mismatch')
    samples = []
    for end, v in zip(endpoints,video):
        motions = [o['motion'] for o in record['frames'][end]['objects'] if o['motion'] is not None]
        samples.append(dict(frame_index=end,G=dino[end].tolist(),T=dino_temporal(dino[end-WINDOW+1:end+1]).tolist(),
            C=motions or None,V=normalize_feature(v).tolist()))
    return dict(id=record['id'],sequence=record['sequence'],split=record['split'],samples=samples)


class TemporalScores:
    def __init__(self, fit, scorer):
        self.scorer = scorer
        self.fit_ids = [r['id'] for r in fit]
        self.banks = {};self.examples = {};self.scales = {};self.thresholds = {}
        for name in ['G','T','C','V']:
            rows = []
            for r in fit:
                for s in r['samples']:
                    if name == 'C':rows.extend(s['C'] or [])
                    else:rows.append(s[name])
            bank = fit_space(rows,4 if name=='C' else 16,standardize=name=='C')
            self.banks[name] = bank
            self.examples[name] = (np.asarray(rows)-bank['center'])/bank['scale']

    def raw(self, records):
        result = []
        for r in records:
            samples = []
            for s in r['samples']:
                out = dict(frame_index=s['frame_index'])
                for name in ['G','T','C','V']:
                    rows = (s['C'] or []) if name=='C' else [s[name]]
                    if not rows:out[name]=None;continue
                    bank = self.banks[name]
                    if self.scorer == 'pca_residual':values=[residual(x,bank) for x in rows]
                    else:values=nearest_distance((np.asarray(rows)-bank['center'])/bank['scale'],self.examples[name])
                    out[name] = max(values)
                samples.append(out)
            result.append(dict(id=r['id'],sequence=r['sequence'],samples=samples))
        return result

    def normalized(self, records):
        scored = self.raw(records)
        for r in scored:
            for s in r['samples']:
                for name in ['G','T','C','V']:
                    if s[name] is not None:s[name]/=self.scales[name]
                s['GV'] = max(s['G'],s['V'])
        return scored

    def calibrate(self, scale, threshold):
        ids = [r['id'] for r in scale+threshold]
        if len(ids)!=len(set(ids)) or set(ids)&set(self.fit_ids):
            raise ValueError('Normal calibration leakage')
        raw = self.raw(scale)
        for name in ['G','T','C','V']:
            values=[s[name] for r in raw for s in r['samples'] if s[name] is not None]
            if len(values)<8:raise ValueError(f'Insufficient normal calibration: {name}')
            self.scales[name]=max(float(np.quantile(values,.99)),1e-6)
        scored = self.normalized(threshold)
        for name in METHODS:
            values=[s[name] for r in scored for s in r['samples'] if s[name] is not None]
            if len(values)<8:raise ValueError(f'Insufficient normal threshold data: {name}')
            self.thresholds[name]=float(np.quantile(values,.99))
        return self


def evaluate_temporal(scored, root, thresholds):
    rows=[];full_frames=0
    for r in scored:
        labels=np.load(root/'test_label'/f"{int(r['sequence']):03}.npy",allow_pickle=False)
        full_frames+=len(labels)
        for s in r['samples']:
            row=dict(sequence=r['sequence'],frame_index=s['frame_index'],label=int(labels[s['frame_index']]))
            row.update({name:s[name] for name in METHODS});rows.append(row)
    shared=[r for r in rows if all(r[k] is not None for k in METHODS)]
    if not shared:raise ValueError('No common scored endpoints')
    models={}
    for name in METHODS:
        valid=[r for r in rows if r[name] is not None]
        per_sequence=[]
        for seq in sorted({r['sequence'] for r in rows}):
            rr=[r for r in shared if r['sequence']==seq]
            per_sequence.append(dict(sequence=seq,frames=len(rr),anomaly_frames=sum(r['label'] for r in rr),
                **rank_metrics([r['label'] for r in rr],[r[name] for r in rr])))
        models[name]=dict(shared_metrics=rank_metrics([r['label'] for r in shared],[r[name] for r in shared]),
            own_sampled_metrics=rank_metrics([r['label'] for r in valid],[r[name] for r in valid]),
            coverage_of_sampled_endpoints=len(valid)/len(rows),threshold=thresholds[name],per_sequence_shared=per_sequence,
            decisions_sampled=decision_metrics([r['label'] for r in valid],[r[name] for r in valid],thresholds[name],
                all_labels=[r['label'] for r in rows]))
    return dict(full_test_frames=full_frames,sampled_endpoints=len(rows),shared_endpoints=len(shared),
        shared_coverage_of_sampled_endpoints=len(shared)/len(rows),sampled_frame_fraction=len(rows)/full_frames,
        shared_anomaly_endpoints=sum(r['label'] for r in shared),models=models,
        note='Endpoint labels only. Unsampled frames have no decisions. Do not compare these AUROCs directly to the previous full-frame table.'),rows


def run_vjepa_cpu():
    project=Path('/content/drive/MyDrive/IPAD_project')
    if not project.is_dir():raise FileNotFoundError('Mount the existing Google Drive before running this script.')
    original=project/'meeting_experiments'/SOURCE_FOLDER
    settings=json.loads((original/'settings.json').read_text(encoding='utf-8'))
    dino_report=json.loads((original/DINO_FOLDER/'dinov2_report.json').read_text(encoding='utf-8'))
    if dino_report['status']!='passed':raise ValueError('DINO run is not complete')
    normal,tested=load_cache(project,settings)
    root=restore_frames(project,settings)
    dino={r['id']:load_dino_global(r,project/'meeting_experiments/dinov2_feature_cache',dino_report['model_signature']) for r in normal+tested}
    for r in tested:
        y=np.load(root/'test_label'/f"{int(r['sequence']):03}.npy",allow_pickle=False)
        if y.shape!=(len(r['frames']),) or not set(np.unique(y))<={0,1}:raise ValueError('Frame label mismatch')
    run=original/('vjepa_cpu_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC'))
    run.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',device='cpu',window=WINDOW,endpoint_step=STEP)
    encoder=None
    try:
        encoder=VideoEncoder();report['model_signature']=encoder.signature
        first=normal[0];paths=source_paths(first,root)
        print('Timing first real pretrained video inference...',flush=True)
        tick=time.perf_counter();vector=encoder(paths[:WINDOW]);elapsed=time.perf_counter()-tick
        count=sum(len(endpoint_indices(len(r['frames']))) for r in normal+tested)
        report['smoke']=dict(seconds=elapsed,feature_dim=len(vector),total_clips=count,
            rough_hours=elapsed*count/3600)
        print(f'First clip: {elapsed:.1f}s. {count} clips total; rough extraction estimate {elapsed*count/3600:.2f} hours, excludes download/scoring.',flush=True)
        # Probe on normal video only; not used for fitting or selecting a score.
        backwards=encoder(list(reversed(paths[:WINDOW])))
        repeated=encoder([paths[WINDOW-1]]*WINDOW)
        report['normal_temporal_probe']=dict(reversed_cosine_distance=float(1-vector@backwards),
            repeated_last_frame_cosine_distance=float(1-vector@repeated),note='Normal-only sensitivity check; not an anomaly detection result.')
        report['notes']=[
            'Official V-JEPA 2 ViT-L/16 pretrained checkpoint, frozen CPU float32 encoder; no predictor anomaly score.',
            'Pretrained for 64 frames at 256; this CPU pilot uses 16 actual past frames at 256 by default.',
            'Full-frame fixed spatial crop, not object-centric V-JEPA. No detector rerun.',
            'G: cached DINO single frame; T: DINO mean + signed endpoint change + mean absolute difference.',
            'C: existing 10-frame object coordinate features; V: mean-pooled V-JEPA video feature; GV: max of normal-scaled G and V.',
            'Same sampled endpoints for comparisons. Common subset additionally requires observed C.',
            'Sampling is fixed before test scoring; endpoint labels, no interpolation or future frames.',
            'Same disjoint normal video groups; fitted observations restricted to sampled endpoints.',
            'Video features mix appearance and time. Better V scores would not by themselves isolate motion contribution.',
            'Different model capacities and input resolution. Exploratory R01 follow-up, not a paper reproduction or novelty claim.']
        write(run/'vjepa_report.json',report)
        records=[];timing=[];cache=project/'meeting_experiments/vjepa_video_cache'
        for i,r in enumerate(normal+tested,1):
            print(f'V-JEPA sequence {i}/49: {r["split"]}/{r["sequence"]}',flush=True)
            video,meta=extract_video_features(r,root,cache,encoder)
            records.append(sampled_features(r,dino[r['id']],video));timing.append(meta)
        encoder.model=None;encoder=None;gc.collect()
        report['timing']=timing;report['variants']={}
        for scorer in ['pca_residual','nearest_normal']:
            print('Evaluating:',scorer,flush=True)
            model=TemporalScores(records[:27],scorer).calibrate(records[27:30],records[30:34])
            metrics,rows=evaluate_temporal(model.normalized(records[34:]),root,model.thresholds)
            report['variants'][scorer]=metrics
            folder=run/scorer;folder.mkdir();write(folder/'metrics.json',metrics)
            with (folder/'frame_scores.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            for name,item in metrics['models'].items():print(name,item['shared_metrics'],flush=True)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,scorer in zip(axes,['pca_residual','nearest_normal']):
            m=report['variants'][scorer]
            values=[m['models'][k]['shared_metrics']['auroc'] for k in METHODS]
            ax.bar(METHODS,[v if v is not None else np.nan for v in values]);ax.set_ylim(0,1)
            ax.set_title(f'{scorer}: {m["shared_endpoints"]} shared endpoints');ax.set_ylabel('AUROC on sampled endpoints')
        fig.tight_layout();fig.savefig(run/'temporal_comparison.png',dpi=160);plt.close(fig)
        report['status']='passed';write(run/'vjepa_report.json',report)
        print('COMPLETE:',run,flush=True)
        print('Share vjepa_report.json and temporal_comparison.png',flush=True)
    except Exception as e:
        import traceback
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());write(run/'vjepa_report.json',report)
        raise
    finally:
        if encoder is not None:encoder.model=None;gc.collect()


if __name__=='__main__':
    prepare_vjepa_dependencies()
    run_vjepa_cpu()
