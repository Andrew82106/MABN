"""CPU-only official model-card example; does not read any project labels."""
import time
import torch
import common as c
import minicheck_official as m

torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
manifest=c.read(c.HERE/'MODEL_MANIFEST.json')
for name,h in manifest['files_sha256'].items():assert c.sha(m.MODEL/name)==h
t=time.time();checker=m.make_inferencer('cpu')
doc='A group of students gather in the school library to study for their upcoming final exams.'
claims=['The students are preparing for an examination.','The students are on vacation.']
pp=[checker.inference_per_example(doc,q) for q in claims]
support=[float(p['max_support_prob']) for p in pp]
assert support[0]>.5>support[1],support
assert not torch.cuda.is_initialized()
c.save(m.OUT/'CPU_MODEL_CHECK.json',{'passed':True,'supported_claim_score':support[0],'unsupported_claim_score':support[1],
  'official_fp32_model_loaded':True,'model_manifest_sha256':c.sha(c.HERE/'MODEL_MANIFEST.json'),
  'elapsed_seconds':time.time()-t,'GPU_used':False,'test_labels_read':False})
print('MINICHECK_CPU_MODEL_CHECK',support,flush=True)
