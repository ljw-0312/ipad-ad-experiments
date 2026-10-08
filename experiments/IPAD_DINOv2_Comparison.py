# Frozen DINOv2-Small / CLIP comparison on existing R01 detections.
# Standalone Colab runner; original cache + source ZIP required.
# Cache-only R01 scoring diagnostic. Execute in the same mounted Colab runtime.
import numpy as np
"""Normal-only PCA and equal-frame evaluation without external metric dependencies."""
import numpy as np


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


class FourModels:
    def __init__(self,state):self.state=state

    @classmethod
    def fit(cls,records,config):
        if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate normal sequence content')
        full=[f['global'] for r in records for f in r['frames']]
        objects=[o['appearance'] for r in records for f in r['frames'] for o in f['objects']]
        motion=[o['motion'] for r in records for f in r['frames'] for o in f['objects'] if o['motion'] is not None]
        if len(objects)<8 or len(motion)<8:
            raise ValueError('Not enough object/motion observations. Inspect normal detections and tracking.')
        return cls(dict(config=config,fit_ids=[r['id'] for r in records],
            banks=dict(A=fit_space(full,config['appearance_rank']),
                       B=fit_space(objects,config['appearance_rank']),
                       C=fit_space(motion,config['motion_rank'],standardize=True)),
            scales={},thresholds={}))

    def raw(self,records):
        out=[]
        for r in records:
            frames=[]
            for f in r['frames']:
                objects=[]
                for o in f['objects']:
                    objects.append(dict(track_id=o['track_id'],bbox=o['bbox'],confidence=o['confidence'],
                        B_raw=residual(o['appearance'],self.state['banks']['B']),
                        C_raw=residual(o['motion'],self.state['banks']['C']) if o['motion'] is not None else None))
                b=[o['B_raw'] for o in objects];c=[o['C_raw'] for o in objects if o['C_raw'] is not None]
                frames.append(dict(frame_index=f['frame_index'],objects=objects,
                    A_raw=residual(f['global'],self.state['banks']['A']),
                    B_raw=max(b) if b else None,C_raw=max(c) if c else None))
            out.append(dict(id=r['id'],sequence=r['sequence'],frames=frames))
        return out

    def normalized(self,raw):
        for r in raw:
            for f in r['frames']:
                for name in ['A','B','C']:
                    f[name]=f[name+'_raw']/self.state['scales'][name] if f[name+'_raw'] is not None else None
                available=[f[k] for k in ['B','C'] if f[k] is not None]
                f['D']=max(available) if available else None
                f['D_partial']=f['B'] is None or f['C'] is None
                for o in f['objects']:
                    o['B']=o['B_raw']/self.state['scales']['B']
                    o['C']=o['C_raw']/self.state['scales']['C'] if o['C_raw'] is not None else None
        return raw

    def calibrate(self,scale_records,threshold_records):
        ids=[r['id'] for r in scale_records+threshold_records]
        if len(ids)!=len(set(ids)) or set(ids)&set(self.state['fit_ids']):raise ValueError('Calibration leakage')
        raw=self.raw(scale_records);q=self.state['config']['quantile']
        for name in ['A','B','C']:
            vals=[f[name+'_raw'] for r in raw for f in r['frames'] if f[name+'_raw'] is not None]
            if len(vals)<8:raise ValueError(f'Insufficient held-out normal calibration for {name}')
            self.state['scales'][name]=max(float(np.quantile(vals,q)),1e-6)
        scored=self.normalized(self.raw(threshold_records))
        for name in ['A','B','C','D']:
            vals=[f[name] for r in scored for f in r['frames'] if f[name] is not None]
            if len(vals)<8:raise ValueError(f'Insufficient held-out normal threshold data for {name}')
            self.state['thresholds'][name]=float(np.quantile(vals,q))
        self.state['scale_ids']=[r['id'] for r in scale_records]
        self.state['threshold_ids']=[r['id'] for r in threshold_records]
        return self

    def score(self,records):
        if not self.state['thresholds']:raise ValueError('Calibrate first')
        return self.normalized(self.raw(records))


import hashlib, json, time, csv, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
NAMES=dict(A='Full frame',B='Object appearance',C='Object motion',D='Object appearance + motion')

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

class NearestNormal(FourModels):
    @classmethod
    def fit(cls,records,config):
        model=super().fit(records,config)
        model.examples=dict(A=np.array([f['global'] for r in records for f in r['frames']]),
            B=np.array([o['appearance'] for r in records for f in r['frames'] for o in f['objects']]),
            C=np.array([o['motion'] for r in records for f in r['frames'] for o in f['objects'] if o['motion'] is not None]))
        for k in model.examples:
            bank=model.state['banks'][k]
            model.examples[k]=(model.examples[k]-bank['center'])/bank['scale']
        return model

    def raw(self,records):
        queries=dict(A=[],B=[],C=[])
        for r in records:
            for f in r['frames']:
                queries['A'].append(f['global'])
                for o in f['objects']:
                    queries['B'].append(o['appearance'])
                    if o['motion'] is not None:queries['C'].append(o['motion'])
        values={}
        for k in queries:
            bank=self.state['banks'][k]
            values[k]=iter(nearest_distance((np.asarray(queries[k])-bank['center'])/bank['scale'],self.examples[k])) if queries[k] else iter([])
        result=[]
        for r in records:
            frames=[]
            for f in r['frames']:
                objects=[dict(track_id=o['track_id'],bbox=o['bbox'],confidence=o['confidence'],B_raw=next(values['B']),
                    C_raw=next(values['C']) if o['motion'] is not None else None) for o in f['objects']]
                b=[o['B_raw'] for o in objects];c=[o['C_raw'] for o in objects if o['C_raw'] is not None]
                frames.append(dict(frame_index=f['frame_index'],objects=objects,A_raw=next(values['A']),
                    B_raw=max(b) if b else None,C_raw=max(c) if c else None))
            result.append(dict(id=r['id'],sequence=r['sequence'],frames=frames))
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

def restore_labels(project,settings,root):
    archive=project/'IPAD_dataset.zip'
    if not archive.is_file():raise FileNotFoundError(f'Cannot restore labels: {archive}')
    with zipfile.ZipFile(archive) as z:
        selected=[];labels={}
        for item in z.infolist():
            if item.is_dir():continue
            parts=Path(item.filename.replace(chr(92),'/')).parts
            if 'R01' not in parts or '__MACOSX' in parts:continue
            rel=parts[parts.index('R01'):]
            is_frame=len(rel)==5 and rel[1] in ['training','testing'] and rel[2]=='frames' and rel[4].endswith('.jpg')
            is_label=len(rel)==3 and rel[1]=='test_label' and rel[2].endswith('.npy')
            if is_frame or is_label:selected.append((Path(*rel).as_posix(),item))
            if is_label:
                if rel[2] in labels:raise ValueError('Duplicate label file')
                labels[rel[2]]=item
        key=hashlib.sha256(json.dumps([(n,i.CRC,i.file_size) for n,i in sorted(selected)]).encode()).hexdigest()
        if key!=settings['data_manifest']:raise ValueError('Dataset ZIP differs from the original experiment')
        if set(labels)!={f'{i:03}.npy' for i in range(1,16)}:raise ValueError('Missing R01 labels')
        target=root/'test_label';target.mkdir(parents=True,exist_ok=True)
        for name,item in labels.items():(target/name).write_bytes(z.read(item))
    print('Restored only 15 label files; no frame extraction or model inference.',flush=True)


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def flatten_scored(scored,root,thresholds):
    rows=[]
    for record in scored:
        y=np.load(Path(root)/'test_label'/f"{int(record['sequence']):03}.npy",allow_pickle=False)
        for f in record['frames']:
            row=dict(sequence=record['sequence'],frame_index=f['frame_index'],label=int(y[f['frame_index']]),
                object_count=len(f['objects']),motion_objects=sum(o['C'] is not None for o in f['objects']),D_partial=f['D_partial'])
            for name in NAMES:
                row[name]=f[name];row[name+'_flag']=f[name]>thresholds[name] if f[name] is not None else None
            rows.append(row)
    return rows

def make_metrics(rows,thresholds):
    shared=[r for r in rows if all(r[k] is not None for k in NAMES)]
    # Comparison to IPAD uses the exact same center target range if all frames are present.
    lengths={seq:max(r['frame_index'] for r in rows if r['sequence']==seq)+1 for seq in set(r['sequence'] for r in rows)}
    center=[r for r in rows if 8<=r['frame_index']<lengths[r['sequence']]-7]
    models={}
    for name in NAMES:
        known=[r for r in rows if r[name] is not None]
        sy=[r['label'] for r in shared];ss=[r[name] for r in shared]
        ky=[r['label'] for r in known];ks=[r[name] for r in known]
        valid_center=[r for r in center if r[name] is not None]
        seq_metrics=[]
        for seq in sorted(lengths):
            rr=[r for r in shared if r['sequence']==seq]
            metric=rank_metrics([r['label'] for r in rr],[r[name] for r in rr])
            seq_metrics.append(dict(sequence=seq,shared_frames=len(rr),anomaly_frames=sum(r['label'] for r in rr),**metric))
        models[name]=dict(name=NAMES[name],threshold=thresholds[name],known_frames=len(known),
            coverage=len(known)/len(rows),shared_frames=len(shared),
            shared_metrics=rank_metrics(sy,ss),own_known_metrics=rank_metrics(ky,ks),
            center_known_metrics=rank_metrics([r['label'] for r in valid_center],[r[name] for r in valid_center]),
            center_known_frames=len(valid_center),
            decisions_all=decision_metrics(ky,ks,thresholds[name],all_labels=[r['label'] for r in rows]),
            shared_decisions=decision_metrics(sy,ss,thresholds[name]),per_sequence_shared=seq_metrics)
    return dict(total_frames=len(rows),shared_frames=len(shared),shared_coverage=len(shared)/len(rows),
        shared_anomaly_frames=sum(r['label'] for r in shared),center_range_frames=len(center),models=models,
        evaluation_notes=['Primary A/B/C/D comparison uses identical shared frames; report coverage to expose detection/track failures.',
            'Own-known metrics use different frame subsets and are not a fair direct comparison.',
            'Recall_all treats unscored anomalous frames as missed. No score-zero imputation.',
            'Thresholds use independent normal scale/threshold groups; no test normalization, no test threshold tuning.',
            'All features are frozen CLIP; this is PCA-inspired exploratory AD, not official SubspaceAD reproduction.',
            'Motion is causal 10 observed contiguous frames; original FPS unknown. No semantic process-order model.'])

def figures(run,rows,metrics):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(16,14))
    for ax,seq in zip(axes.flat,sorted(set(r['sequence'] for r in rows))):
        rr=[r for r in rows if r['sequence']==seq];idx=[r['frame_index'] for r in rr]
        for name in NAMES:
            ax.plot(idx,[r[name] if r[name] is not None else np.nan for r in rr],label=name,linewidth=.8)
        ax.fill_between(idx,0,1,where=np.array([r['label'] for r in rr],bool),transform=ax.get_xaxis_transform(),color='red',alpha=.12)
        ax.set_title(seq);ax.set_xlabel('frame index')
    axes.flat[0].legend();fig.tight_layout();fig.savefig(Path(run)/'score_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    vals=[metrics['models'][k]['shared_metrics']['auroc'] for k in NAMES]
    ax.bar(list(NAMES),[x if x is not None else np.nan for x in vals]);ax.set_ylim(0,1)
    for i,x in enumerate(vals):
        if x is None:ax.text(i,.02,'N/A',ha='center')
    ax.set_ylabel('AUROC on identical shared frames');ax.set_title(f"Coverage: {metrics['shared_coverage']:.1%}")
    fig.tight_layout();fig.savefig(Path(run)/'comparison.png',dpi=150);plt.close(fig)
def run_cached_diagnostic():
    # Numerical check: nearest neighbour must match exhaustive direct distances.
    rng=np.random.default_rng(42);q=rng.normal(size=(9,7));b=rng.normal(size=(13,7))
    direct=np.sqrt(np.min(np.sum((q[:,None]-b[None,:])**2,axis=2),axis=1)/7)
    np.testing.assert_allclose(nearest_distance(q,b,chunk=3),direct,rtol=1e-5,atol=1e-6)
    project=Path('/content/drive/MyDrive/IPAD_project')
    if not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    original=project/'meeting_experiments/R01_20261007_102640_886381_UTC'
    if not (original/'settings.json').is_file():
        candidates=sorted((project/'meeting_experiments').glob('R01_*/settings.json'))
        raise FileNotFoundError('Original result folder unavailable. Visible settings files: '+str([str(p) for p in candidates]))
    settings=json.loads((original/'settings.json').read_text(encoding='utf-8'))
    prior=json.loads((original/'meeting_report.json').read_text(encoding='utf-8'))
    root=Path('/content/meeting_ad_data')/settings['data_manifest'][:16]/'R01'
    if not all((root/'test_label'/f'{i:03}.npy').is_file() for i in range(1,16)):
        restore_labels(project,settings,root)
    run=original/('cached_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    run.mkdir(parents=True,exist_ok=False)
    print('Cache-only scoring. No detector, encoder, or GPU inference.',flush=True)
    normal,tested=load_cache(project,settings);summary={};table=[]
    for variant,kind in [('pca_residual',FourModels),('nearest_normal',NearestNormal)]:
        tick=time.perf_counter();print('Scoring:',variant,flush=True)
        model=kind.fit(normal[:27],settings['config']).calibrate(normal[27:30],normal[30:])
        rows=flatten_scored(model.score(tested),root,model.state['thresholds'])
        metrics=make_metrics(rows,model.state['thresholds'])
        if variant=='pca_residual':
            for name in NAMES:
                a=metrics['models'][name]['shared_metrics']['auroc'];old=prior['metrics']['models'][name]['shared_metrics']['auroc']
                if not np.isclose(a,old,rtol=0,atol=1e-10):raise ValueError('Baseline mismatch: verify caches and source config')
        folder=run/variant;folder.mkdir()
        write(folder/'metrics.json',metrics);figures(folder,rows,metrics)
        with (folder/'frame_scores.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary[variant]=dict(seconds=time.perf_counter()-tick,metrics=metrics)
        for name,item in metrics['models'].items():
            row=dict(variant=variant,model=name,auroc=item['shared_metrics']['auroc'],
                ap=item['shared_metrics']['average_precision'],coverage=item['coverage'],
                recall_all=item['decisions_all']['recall_all'],normal_fpr=item['decisions_all']['normal_frame_false_positive_rate_known'])
            table.append(row);print(row,flush=True)
    write(run/'diagnostic_report.json',dict(status='passed',source_run=str(original),variants=summary,
        notes=['Same cached features, tracks, shared frames and normal-only partitions.',
            'Nearest normal uses all fit examples; no test labels used for fitting/calibration.',
            'Follow-up motivated by observed R01 results: exploratory, not independent confirmation.',
            'Nearest neighbour is a diagnostic baseline, not a claimed novel method.']))
    with (run/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print('COMPLETE:',run,flush=True)
    print('Share diagnostic_report.json and nearest_normal/comparison.png',flush=True)
    return run


import copy
import gc
import importlib.util
import math
import os
import subprocess
import sys
import time
from PIL import Image

SOURCE_FOLDER = 'R01_20261007_102640_886381_UTC'
DINO_MODEL = 'facebook/dinov2-small'


def choose_device(requested, cuda_available):
    if requested not in ['auto', 'cpu', 'cuda']:
        raise ValueError('IPAD_DEVICE must be auto, cpu or cuda; TPU needs a separate runner.')
    if requested == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA requested but no NVIDIA GPU is available.')
    return 'cuda' if requested == 'cuda' or (requested == 'auto' and cuda_available) else 'cpu'


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


def feature_arrays(record, full, objects, dim):
    full = np.asarray(full, dtype=np.float32)
    objects = np.asarray(objects, dtype=np.float32).reshape(-1, dim)
    expected = sum(len(f['objects']) for f in record['frames'])
    if full.shape != (len(record['frames']), dim) or objects.shape != (expected, dim):
        raise ValueError('DINO feature/frame/object count mismatch')
    if not np.isfinite(full).all() or not np.isfinite(objects).all():
        raise ValueError('Nonfinite DINO features')
    for array in [full, objects]:
        if len(array) and not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError('DINO features must be L2 normalized')
    return full, objects


def replace_appearance(record, full, objects, dim):
    full, objects = feature_arrays(record, full, objects, dim)
    result = copy.deepcopy(record)
    row = 0
    for f, z in zip(result['frames'], full):
        f['global'] = z.tolist()
        for o in f['objects']:
            o['appearance'] = objects[row].tolist()
            row += 1
    return result


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


def image_crops(path, objects):
    with Image.open(path) as im:
        image = im.convert('RGB')
    images = [image]
    for o in objects:
        x1, y1, x2, y2 = o['bbox']
        dx, dy = (x2-x1)*.05, (y2-y1)*.05
        images.append(image.crop((max(0, x1-dx), max(0, y1-dy), min(image.width, x2+dx), min(image.height, y2+dy))))
    return images


def extract_dino(record, root, cache_dir, signature, encode, batch_frames=8):
    """Encode the same frames and crops in fixed order, cache per sequence."""
    paths = source_paths(record, root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((record['signature'] + json.dumps(signature, sort_keys=True)).encode()).hexdigest()
    target = cache_dir / (key + '.npz')
    dim = signature['feature_dim']
    if target.is_file():
        with np.load(target, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['signature'] != signature or metadata['source_id'] != record['id']:
                raise ValueError('DINO cache signature mismatch')
            result = replace_appearance(record, data['global'], data['objects'], dim)
        metadata['cache_hit'] = True
        print(f"Cached DINO: {record['split']}/{record['sequence']}", flush=True)
        return result, metadata
    from tqdm.auto import tqdm
    globals_, object_features = [], []
    tick = time.perf_counter()
    progress = tqdm(range(0, len(paths), batch_frames), desc=f"DINO {record['split']}/{record['sequence']}", leave=False)
    for start in progress:
        group = record['frames'][start:start+batch_frames]
        images, counts = [], []
        for f, path in zip(group, paths[start:start+batch_frames]):
            crops = image_crops(path, f['objects'])
            images.extend(crops)
            counts.append(len(crops))
        z = np.asarray(encode(images), dtype=np.float32)
        if z.shape != (sum(counts), dim):
            raise ValueError('Encoder output shape mismatch')
        at = 0
        for count in counts:
            globals_.append(z[at])
            object_features.extend(z[at+1:at+count])
            at += count
        for im in images:
            im.close()
    seconds = time.perf_counter()-tick
    full, obj = feature_arrays(record, globals_, object_features, dim)
    metadata = dict(signature=signature, source_id=record['id'], sequence=record['sequence'], split=record['split'],
                    frames=len(paths), seconds=seconds, frames_per_second=len(paths)/seconds, cache_hit=False,
                    execution_device=getattr(encode, 'device', 'test'),
                    timing_scope='local JPG reads, crop, processor, DINO inference and CPU feature transfer; excludes source hashing and cache write')
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **{'global': full, 'objects': obj, 'metadata': np.array(json.dumps(metadata))})
    temporary.replace(target)
    print(f"DINO {record['split']}/{record['sequence']}: {len(paths)} frames in {seconds:.1f}s ({len(paths)/seconds:.2f} frames/s)", flush=True)
    return replace_appearance(record, full, obj, dim), metadata


class DinoEncoder:
    def __init__(self):
        import torch
        self.device = choose_device(os.environ.get('IPAD_DEVICE', 'auto').lower(), torch.cuda.is_available())
        if self.device == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        print(f'DINO inference device: {self.device}', flush=True)
        from transformers import AutoConfig, AutoImageProcessor, Dinov2Model
        self.torch = torch
        config = AutoConfig.from_pretrained(DINO_MODEL)
        revision = getattr(config, '_commit_hash', None)
        if not revision:
            raise RuntimeError('Could not resolve DINO model revision')
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL, revision=revision, use_fast=False)
        self.model = Dinov2Model.from_pretrained(DINO_MODEL, revision=revision, config=config).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.signature = dict(model=DINO_MODEL, revision=revision, feature_dim=config.hidden_size,
            feature='last_hidden_state CLS token; L2 normalized', precision='float32',
            preprocessing='shortest side 224, center crop 224, bicubic; DINO ImageNet normalization',
            processor_revision=revision, extraction_version=1, crop_margin=.05)
        self.batch_images = 32 if self.device == 'cuda' else 8

    def __call__(self, images):
        torch = self.torch
        chunks = []
        start = 0
        while start < len(images):
            count = min(self.batch_images, len(images)-start)
            try:
                with torch.inference_mode():
                    inputs = self.processor(images=images[start:start+count], return_tensors='pt',
                        size={'shortest_edge': 224}, crop_size={'height': 224, 'width': 224},
                        do_resize=True, do_center_crop=True).to(self.device)
                    output = self.model(**inputs)
                    z = torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
                chunks.append(z)
                del inputs, output, z
                start += count
            except torch.cuda.OutOfMemoryError:
                if self.device != 'cuda' or count == 1:
                    raise
                # Drop partially allocated tensors before a smaller retry.
                inputs = output = None
                gc.collect()
                torch.cuda.empty_cache()
                self.batch_images = max(1, count//2)
                print(f'GPU batch reduced to {self.batch_images}; no frame is skipped.', flush=True)
        return np.concatenate(chunks)

    def close(self):
        self.model = None
        gc.collect()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()


def ensure_dependencies():
    packages = {'transformers': 'transformers==4.52.2', 'matplotlib': 'matplotlib', 'tqdm': 'tqdm', 'PIL': 'Pillow'}
    missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    # Existing torch / CUDA and working transformers are retained.


def save_evaluation(run, variant, model, tested, root):
    rows = flatten_scored(model.score(tested), root, model.state['thresholds'])
    metrics = make_metrics(rows, model.state['thresholds'])
    folder = run / variant
    folder.mkdir()
    write(folder/'metrics.json', metrics)
    figures(folder, rows, metrics)
    with (folder/'frame_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics, rows


def assert_same_observations(clip_rows, dino_rows):
    if len(clip_rows) != len(dino_rows):
        raise ValueError('Different frame counts')
    for a, b in zip(clip_rows, dino_rows):
        for key in ['sequence', 'frame_index', 'label', 'object_count', 'motion_objects', 'D_partial']:
            if a[key] != b[key]:
                raise ValueError(f'Observation mismatch: {key}')
        for name in NAMES:
            if (a[name] is None) != (b[name] is None):
                raise ValueError('Different evaluation subsets')
        if a['C'] is not None and not np.isclose(a['C'], b['C'], rtol=0, atol=1e-10):
            raise ValueError('Motion control changed unexpectedly')


def run_dinov2_comparison():
    project = Path('/content/drive/MyDrive/IPAD_project')
    if not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    original = project/'meeting_experiments'/SOURCE_FOLDER
    if not (original/'settings.json').is_file():
        raise FileNotFoundError(f'Original result folder unavailable: {original}')
    settings = json.loads((original/'settings.json').read_text(encoding='utf-8'))
    prior = json.loads((original/'meeting_report.json').read_text(encoding='utf-8'))
    ensure_dependencies()
    print('Loading saved CLIP features, boxes and tracks. No GroundingDINO execution.', flush=True)
    normal, tested = load_cache(project, settings)
    partitions=[settings['fit_sequences'],settings['scale_calibration_sequences'],settings['threshold_calibration_sequences']]
    if partitions!=[[r['sequence'] for r in normal[:27]],[r['sequence'] for r in normal[27:30]],[r['sequence'] for r in normal[30:]]]:
        raise ValueError('Source normal partitions differ from this controlled comparison')
    root = restore_frames(project, settings)
    for r in tested:
        y = np.load(root/'test_label'/f"{int(r['sequence']):03}.npy", allow_pickle=False)
        if y.shape != (len(r['frames']),) or not set(np.unique(y)) <= {0, 1}:
            raise ValueError('Frame labels invalid')
    run = original / ('dinov2_comparison_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC'))
    run.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', source_run=str(original), config=settings['config'], variants={})
    encoder = None
    try:
        import torch
        torch.manual_seed(42)
        encoder = DinoEncoder()
        first = normal[0]
        smoke_images = image_crops(root/'training/frames'/first['sequence']/'000.jpg', first['frames'][0]['objects'])
        smoke = encoder(smoke_images)
        if smoke.shape != (len(smoke_images), encoder.signature['feature_dim']) or not np.isfinite(smoke).all():
            raise ValueError('DINO smoke inference failed')
        for im in smoke_images:
            im.close()
        if encoder.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        report.update(model_signature=encoder.signature, execution_device=encoder.device,
            gpu=torch.cuda.get_device_name() if encoder.device == 'cuda' else None,
            cpu_threads=torch.get_num_threads(),
            preprocessing_note='DINO resize/crop geometry matches prior CLIP 224 setup; normalization is model-specific. Native DINO shortest-side 256 was overridden to 224.',
            partitions=dict(fit=settings['fit_sequences'],scale=settings['scale_calibration_sequences'],threshold=settings['threshold_calibration_sequences']),
            notes=['Frozen DINOv2-Small CLS features vs frozen CLIP ViT-B/32.',
                'Original boxes, tracks, crop margin, frame order and motion descriptors are reused.',
                'PCA and nearest-normal scorers are each held fixed across encoders.',
                'All thresholds use disjoint normal data, no test labels for fitting.',
                'CPU/CUDA use the same float32 extraction recipe and caches; minor numerical device differences are possible.',
                'Single-scene exploratory follow-up; not an independent confirmation or paper reproduction.',
                'C is unchanged motion control; D is exploratory fusion. Primary encoder comparison is A and B.'])
        write(run/'settings.json', report)
        dino_records, timing = [], []
        cache = project/'meeting_experiments/dinov2_feature_cache'
        records = normal+tested
        observations = len(records)
        for i, r in enumerate(records, 1):
            print(f'DINO sequence {i}/{observations}: {r["split"]}/{r["sequence"]}', flush=True)
            new, meta = extract_dino(r, root, cache, encoder.signature, encoder)
            dino_records.append(new)
            timing.append(meta)
            if i == 1 and not meta['cache_hit']:
                remaining = sum(len(x['frames']) for x in records[1:])
                print(f'Estimated remaining extraction: {remaining/meta["frames_per_second"]/60:.1f} minutes; excludes I/O variation and scoring.', flush=True)
        report['peak_gpu_gib'] = torch.cuda.max_memory_allocated()/1024**3 if encoder.device == 'cuda' else None
        report['extraction_timing'] = timing
        report['effective_batch_images'] = encoder.batch_images
        encoder.close()
        encoder = None
        table = []
        for scorer, kind in [('pca_residual', FourModels), ('nearest_normal', NearestNormal)]:
            reference_rows = None
            for name, training, testing in [('clip', normal, tested), ('dinov2', dino_records[:34], dino_records[34:])]:
                variant = name+'_'+scorer
                print('Evaluating:', variant, flush=True)
                model = kind.fit(training[:27], settings['config']).calibrate(training[27:30], training[30:])
                metrics, rows = save_evaluation(run, variant, model, testing, root)
                if name == 'clip':
                    reference_rows = rows
                    if scorer == 'pca_residual':
                        for k in NAMES:
                            if not np.isclose(metrics['models'][k]['shared_metrics']['auroc'],
                                prior['metrics']['models'][k]['shared_metrics']['auroc'], rtol=0, atol=1e-10):
                                raise ValueError('Original CLIP PCA results did not reproduce')
                else:
                    assert_same_observations(reference_rows, rows)
                report['variants'][variant] = metrics
                for k, item in metrics['models'].items():
                    row = dict(encoder=name,scorer=scorer,model=k,auroc=item['shared_metrics']['auroc'],
                        ap=item['shared_metrics']['average_precision'],coverage=item['coverage'],
                        recall_all=item['decisions_all']['recall_all'],normal_fpr=item['decisions_all']['normal_frame_false_positive_rate_known'])
                    table.append(row)
                    print(row, flush=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, scorer in zip(axes, ['pca_residual', 'nearest_normal']):
            x = np.arange(4)
            for offset, name in [(-.18, 'clip'), (.18, 'dinov2')]:
                values = [report['variants'][name+'_'+scorer]['models'][k]['shared_metrics']['auroc'] for k in NAMES]
                ax.bar(x+offset, values, width=.36, label=name)
            ax.set_xticks(x, list(NAMES));ax.set_ylim(0, 1);ax.set_title(scorer)
            ax.set_ylabel('AUROC, identical shared frames');ax.legend()
        fig.tight_layout();fig.savefig(run/'encoder_comparison.png', dpi=160);plt.close(fig)
        with (run/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader();writer.writerows(table)
        report['status'] = 'passed'
        write(run/'dinov2_report.json', report)
        print('COMPLETE:', run, flush=True)
        print('Share dinov2_report.json and encoder_comparison.png', flush=True)
        from IPython.display import display
        display(Image.open(run/'encoder_comparison.png'))
        return run
    except Exception as e:
        import traceback
        report.update(status='failed',error=str(e),traceback=traceback.format_exc())
        write(run/'dinov2_report.json', report)
        raise
    finally:
        if encoder is not None:
            encoder.close()


if __name__ == '__main__':
    DINO_RUN = run_dinov2_comparison()
