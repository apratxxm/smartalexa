import json, sys, numpy as np
from audio_io import read_wav_mono16k, SR
import wakelib as W
from eval import load_model, clip_scores
sm=json.load(open('split_manifest.json'))
pos=[p for p in sm['positives'] if p['split']=='train']
order=np.random.default_rng(1337).permutation(len(pos))[:14]
vp=[pos[i] for i in order]
mp=sys.argv[1]
m,key=load_model(mp)
rng=np.random.default_rng(5)
bank=[read_wav_mono16k(p['path']) for p in pos]
W.ROOM_BANK.extend(W.build_room_bank(bank))
for mode in ('zeros','room'):
    mx=[];pk=[]
    for p in vp:
        x=read_wav_mono16k(p['path'])[:int(2.5*SR)]
        lead = np.zeros(2*SR,np.float32) if mode=='zeros' else W.room_pad(2*SR,rng)
        tail = np.zeros(SR,np.float32) if mode=='zeros' else W.room_pad(SR,rng)
        s=clip_scores(m,key,np.concatenate([lead,x,tail]))
        mx.append(s.max()); pk.append(int(s.argmax()))
    print(mode,'max',np.round(mx,2),'det@.3/.5/.7',[float(np.mean(np.array(mx)>=t)) for t in (.3,.5,.7)],'argmax chunk',pk)
