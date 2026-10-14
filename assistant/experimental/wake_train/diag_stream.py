"""Streaming sanity check on INNER-VAL clips (taken from TRAIN split; not the held-out set)."""
import json, hashlib, sys, numpy as np
from audio_io import read_wav_mono16k, SR
import wakelib as W
from eval import load_model, clip_scores, n_activations
sm=json.load(open('split_manifest.json'))
pos=[p for p in sm['positives'] if p['split']=='train']
order=np.random.default_rng(1337).permutation(len(pos))[:14]
vp=[pos[i] for i in order]
hp=lambda k:int(hashlib.md5(k.encode()).hexdigest(),16)%100
vn=[n for n in sm['negatives_short'] if n['split']=='train' and hp('short'+n['path'])<15]
mp=sys.argv[1] if len(sys.argv)>1 else r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx"
m,key=load_model(mp)
mx=[]
for p in vp:
    x=read_wav_mono16k(p['path'])[:int(2.5*SR)]
    mx.append(clip_scores(m,key,x).max())
print('inner-val pos max scores', np.round(mx,2), 'det@.3/.5/.7', [float(np.mean(np.array(mx)>=t)) for t in (.3,.5,.7)])
fa={t:0 for t in (.3,.5,.7)}; hrs=0
for n in vn:
    x=read_wav_mono16k(n['path']); s=clip_scores(m,key,x); hrs+=len(x)/SR/3600
    for t in fa: fa[t]+=n_activations(s,t)
print('val neg clips',len(vn),'hours',round(hrs,4),'acts',fa)
