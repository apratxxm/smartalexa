import json
r=json.load(open('A:/pisha/memoir/assistant/models/wake/alexa_user_results.json'))
d=r['data']; print('neg hours local',round(d['holdout_negative_hours_total_local'],3),'generic',round(d['generic_holdout_hours'],2), 'pos old',d['holdout_positive_clips'])
for m in ('custom','stock'):
    print('=====',m)
    for pat in (1,2,3):
        R=r[m][f'patience_{pat}']
        for t in ('0.3','0.5','0.7'):
            L=R['local_negatives']
            print(f"p{pat} t{t}: old38 {R['detection_room_lead'][t]:.2f} (zero {R['detection_zero_lead'][t]:.2f}, music5dB {R['detection_room_lead_music_snr_5dB'][t]:.2f}) | enroll_hold all {R['detection_enroll_holdout_all'][t]:.2f} quiet {R['detection_enroll_holdout_quiet'][t]:.2f} music {R['detection_enroll_holdout_music'][t]:.2f} | rec40s {R['music_recording_40s'][t]['detected_of_19']}/19 FA {R['music_recording_40s'][t]['false_activations']} | generic/h {R['generic_negatives']['per_hour'][t]:.1f} | local acts {L['ALL_LOCAL']['activations'][t]} adv {R['adversarial_word_false_trigger_rate'][t]:.2f}")
