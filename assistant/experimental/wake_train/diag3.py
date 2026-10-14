import json, numpy as np
from audio_io import read_wav_mono16k, SR
from eval import load_model, clip_scores
sm=json.load(open('split_manifest.json'))
mc,kc=load_model(r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx")
n=[n for n in sm['negatives_short'] if n['source']=='retry_audio_path'][3]
x=read_wav_mono16k(n['path'])
s=clip_scores(mc,kc,x)
print(len(x)/SR, np.round(s,2))
z=np.zeros(5*SR,np.float32); print('zeros',np.round(clip_scores(mc,kc,z),2)[:30])
r=(np.random.default_rng(0).standard_normal(5*SR)*0.01).astype(np.float32); print('noise',np.round(clip_scores(mc,kc,r),2)[:30])
