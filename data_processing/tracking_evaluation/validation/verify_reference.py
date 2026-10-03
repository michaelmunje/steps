"""Isolated differential check; official source files remain unmodified."""
from pathlib import Path
import importlib.util
import json
import math
import random
import sys
import types
from unittest.mock import patch
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parent
OFFICIAL=ROOT/'official'
sys.path.insert(0,'/home/zhengpengen/gdc_atrium')
from tracking_evaluation import metrics
# Compatibility shims for official legacy dtype aliases, in this harness only.
if not hasattr(np,'float'): np.float=float
if not hasattr(np,'int'): np.int=int
for name,path in [('reference_trackeval',OFFICIAL/'trackeval'),('reference_trackeval.metrics',OFFICIAL/'trackeval/metrics')]:
 module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
clock=types.ModuleType('reference_trackeval._timing');clock.time=lambda function:function
sys.modules[clock.__name__]=clock
utils=types.ModuleType('reference_trackeval.utils')
utils.TrackEvalException=RuntimeError
utils.init_config=lambda config,defaults,name:{**defaults,**(config or {})}
sys.modules[utils.__name__]=utils
from reference_trackeval.metrics.hota import HOTA
from reference_trackeval.metrics.identity import Identity

def convert(seq,config):
 gids=sorted({n['track_id'] for f in seq['frames'] for n in f['gt']})
 pids=sorted({n['track_id'] for f in seq['frames'] for n in f['pred']})
 gi={name:i for i,name in enumerate(gids)};pi={name:i for i,name in enumerate(pids)}
 data={'num_gt_ids':len(gids),'num_tracker_ids':len(pids),'num_gt_dets':sum(len(f['gt']) for f in seq['frames']),
       'num_tracker_dets':sum(len(f['pred']) for f in seq['frames']),'num_timesteps':len(seq['frames']),
       'gt_ids':[],'tracker_ids':[],'similarity_scores':[]}
 binary=[]
 for frame in seq['frames']:
  gt=sorted(frame['gt'],key=lambda n:n['track_id']);pred=sorted(frame['pred'],key=lambda n:n['track_id'])
  data['gt_ids'].append(np.array([gi[n['track_id']] for n in gt],dtype=int))
  data['tracker_ids'].append(np.array([pi[n['track_id']] for n in pred],dtype=int))
  similarities=np.zeros((len(gt),len(pred)));eligible=np.zeros_like(similarities)
  for row,g in enumerate(gt):
   for column,p in enumerate(pred):
    distance=math.hypot(g['x']-p['x'],g['y']-p['y'])
    if distance<=config['max_distance_m']:
     eligible[row,column]=1
     relative=distance/config['similarity_scale_m']
     similarities[row,column]=max(0,1-relative) if config['similarity']=='linear' else math.exp(-0.5*relative*relative)
  data['similarity_scores'].append(similarities);binary.append(eligible)
 return data,{**data,'similarity_scores':binary}

rng=random.Random(20260914)
sequences=[]
gt_names=['0','001','GT:alpha /A','person Z','source_id:8']
pred_names=['0','001','camera 1/A','q:delta','shared Y','007','p/Z']
for index in range(30):
 frames=[]
 for time in range(rng.randrange(1,10)):
  gt=[{'track_id':name,'x':rng.uniform(-2,2),'y':rng.uniform(-2,2),'yaw':None}
      for name in rng.sample(gt_names,rng.randrange(6))]
  pred=[]
  for name in rng.sample(pred_names,rng.randrange(8)):
   anchor=rng.choice(gt) if gt and rng.random()<0.7 else {'x':rng.uniform(-2,2),'y':rng.uniform(-2,2)}
   pred.append({'track_id':name,'x':anchor['x']+rng.uniform(-0.6,0.6),'y':anchor['y']+rng.uniform(-0.6,0.6),'yaw':None})
  frames.append({'frame_index':time*4,'timestamp_s':time/5,'gt':gt,'pred':pred})
 sequences.append({'name':f'random-{index}','frames':frames})
# Exercise both one-sided and fully empty cases in the differential aggregate.
sequences += [{'name':'empty','frames':[]},
 {'name':'gt-only','frames':[{'frame_index':0,'gt':[{'track_id':'g','x':0,'y':0}],'pred':[]}]},
 {'name':'pred-only','frames':[{'frame_index':0,'gt':[],'pred':[{'track_id':'p','x':0,'y':0}]}]}]
maximum_error=0.0;checked_values=0

def compare(ours,hota,identity):
 global maximum_error,checked_values
 for field in ['HOTA_TP','HOTA_FP','HOTA_FN','HOTA','DetA','AssA']:
  left=np.array([row[field] for row in ours['hota_per_alpha']])
  right=hota[field]
  np.testing.assert_allclose(left,right,rtol=0,atol=1e-12,err_msg=ours['name']+'/'+field)
  maximum_error=max(maximum_error,float(np.max(np.abs(left-right))))
  checked_values+=len(left)
 for field in ['HOTA','DetA','AssA']:
  delta=abs(ours[field]-float(np.mean(hota[field])))
  assert delta<1e-12,(ours['name'],field,delta)
  maximum_error=max(maximum_error,delta);checked_values+=1
 for field in ['IDTP','IDFP','IDFN']:
  assert ours[field]==int(identity[field]),(ours['name'],field,ours[field],identity[field])
  checked_values+=1
 if ours['IDF1'] is None:
  assert ours['total_GT_observations']+ours['total_prediction_observations']==0
  assert identity['IDF1']==0  # Documented null-versus-zero undefined convention.
 else:
  delta=abs(ours['IDF1']-float(identity['IDF1']));assert delta<1e-12
  maximum_error=max(maximum_error,delta)
 checked_values+=1

checks=[]
for kind in ['linear','gaussian']:
 config=metrics.normalize_config({'max_distance_m':1.2,'similarity':kind,'similarity_scale_m':1.4})
 hota_metric=HOTA();id_metric=Identity({'THRESHOLD':0.5,'PRINT_CONFIG':False})
 references={}
 for seq in sequences:
  data,identity_data=convert(seq,config)
  references[seq['name']]=(hota_metric.eval_sequence(data),id_metric.eval_sequence(identity_data))
 combined_hota=hota_metric.combine_sequences({name:value[0] for name,value in references.items()})
 combined_identity=id_metric.combine_sequences({name:value[1] for name,value in references.items()})
 for backend in ['stdlib_hungarian','scipy']:
  selected=None if backend=='stdlib_hungarian' else scipy.optimize.linear_sum_assignment
  with patch.object(metrics,'_scipy_assignment',selected):
   actual=[]
   for seq in sequences:
    result=metrics.evaluate_sequence(seq,config);actual.append(result)
    assert result['assignment_backend']==backend
    compare(result,*references[seq['name']])
   aggregate=metrics.aggregate_results(actual)
   compare(aggregate,combined_hota,combined_identity)
   json.dumps(aggregate,allow_nan=False)
   checks.append({'similarity':kind,'backend':backend,'sequences':len(actual),'aggregate_checked':True,
                  'aggregate_HOTA':aggregate['HOTA'],'aggregate_IDF1':aggregate['IDF1']})
report={'status':'passed','random_seed':20260914,'random_sequences':30,'edge_case_sequences':3,
        'alphas':19,'sequence_comparisons':len(checks)*len(sequences),'aggregate_comparisons':len(checks),
        'scalar_values_checked':checked_values,'maximum_absolute_difference':maximum_error,
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'reference':json.loads((OFFICIAL/'provenance.json').read_text()),
        'reference_shims':['identity timing decorator','default/config merge without printing','legacy np.float/np.int aliases'],
        'undefined_convention':'empty IDF1 is null in our output and zero in TrackEval; all defined values compared',
        'checks':checks}
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
