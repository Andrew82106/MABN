"""CPU algebra/gradient check only; no model or real training data."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F


def aggregate(logits,valid_docs):
    assert logits.ndim==2 and valid_docs.shape==(logits.shape[0],) and valid_docs.any()
    return logits.masked_fill(~valid_docs[:,None],float('inf')).min(dim=0).values


if __name__=='__main__':
    torch.set_num_threads(4)
    # Each of two tokens is supported by a different document.
    z=torch.tensor([[-2.,2.],[2.,-2.]],requires_grad=True);mask=torch.tensor([True,True]);combined=aggregate(z,mask)
    assert torch.equal(combined,torch.tensor([-2.,-2.]))
    assert torch.equal(torch.sigmoid(combined),torch.sigmoid(z).min(dim=0).values)
    loss=F.binary_cross_entropy_with_logits(combined,torch.zeros(2));loss.backward()
    assert z.grad[0,0]>0 and z.grad[1,1]>0 and z.grad[0,1]==0 and z.grad[1,0]==0
    # A fake padded doc cannot win the minimum, and ties deterministically choose the first real doc.
    assert torch.equal(aggregate(torch.cat((z.detach(),torch.full((1,2),-99.))),torch.tensor([True,True,False])),combined.detach())
    tied=torch.zeros(2,1,requires_grad=True);aggregate(tied,mask).sum().backward();assert tied.grad.tolist()==[[1.],[0.]]
    # For a hallucinated token the currently most erroneously supportive doc is penalized.
    h=torch.tensor([[-2.],[2.]],requires_grad=True);F.binary_cross_entropy_with_logits(aggregate(h,mask),torch.ones(1)).backward()
    assert h.grad[0,0]<0 and h.grad[1,0]==0
    result={'passed':True,'supported_tokens_may_choose_different_docs':True,'min_risk_equivalent_max_support':True,
        'gold_only_after_document_aggregation':True,'padded_document_excluded':True,'tie_gradient_first_real_doc':True,
        'real_models_trained':False,'GPU_used':False,'formal_training_protocol_frozen':False}
    Path(__file__).with_name('AGGREGATION_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result)
