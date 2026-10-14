import json, hashlib, numpy as np
from audio_io import read_wav_mono16k, SR
from eval import load_model, clip_scores
sm=json.load(open('split_manifest.json'))
mc,kc=load_model(r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx")
ms,ks=load_model(r"A:\wake-data\models\alexa_v0.1.onnx")
for n in sm['negatives_short']:
    x=read_wav_mono16k(n['path'])
    print(n['source'][:12],n['split'],round(len(x)/SR,1),'custom',round(float(clip_scores(mc,kc,x).max()),2),'stock',round(float(clip_scores(ms,ks,x).max()),2),repr(n['transcript'][:40]))
