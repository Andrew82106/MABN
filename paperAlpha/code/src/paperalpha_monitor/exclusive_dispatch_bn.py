"""Categorical dispatch slots composed with existing shared policy fragments.

Each slot chooses none or one concrete candidate exactly once within the stated
window. The runtime contract supplies exclusivity; a paper, readable role name,
or the agent promising to buy one item does not establish it. Distinct slots
remain conditionally independent here; shared authority is still joint.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np
from scipy.special import logsumexp

from .workflow_risk_bn import WorkflowRiskInput,WorkflowRiskBN


class ExclusiveDispatchInput(WorkflowRiskInput):
    def __init__(self, *, choice_groups, **kwargs):
        self.choice_groups=copy.deepcopy(choice_groups)
        super().__init__(**kwargs)
        self._check_groups()

    def _check_groups(self):
        if not isinstance(self.choice_groups,list) or not self.choice_groups:
            raise ValueError("explicit nonempty exclusive dispatch slots required")
        if any(v!="tool_dispatch" for v in self.event_semantics.values()):
            raise ValueError("exclusive dispatch slots cannot silently reinterpret text/proposal events")
        seen=[]; names=set(); proposals={p["action_id"]:p for p in self.proposals}
        for group in self.choice_groups:
            if not isinstance(group,dict) or set(group)!={"id","action_ids","rule","evidence"}:
                raise ValueError("slot identity, alternatives, rule and runtime evidence required")
            if not isinstance(group["id"],str) or not group["id"] or group["id"] in names:
                raise ValueError("unique nonempty slot identities required")
            names.add(group["id"])
            ids=group["action_ids"]
            if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(proposals):
                raise ValueError("nonempty unique known candidate actions required")
            if group["rule"]!="at_most_one_dispatch_in_window":
                raise ValueError("supported enforced runtime slot rule required")
            if len({(proposals[a]["node"],proposals[a]["binding"]["actor"]) for a in ids})!=1:
                raise ValueError("a dispatch slot belongs to one actor and functional node")
            evidence=group["evidence"]
            if not isinstance(evidence,dict) or set(evidence)!={"record_id","sequence","source","text"}:
                raise ValueError("runtime contract evidence required")
            if type(evidence["sequence"]) is not int or not 0<=evidence["sequence"]<=self.cutoff:
                raise ValueError("slot contract must be available at the forecast cutoff")
            if any(not isinstance(evidence[k],str) or not evidence[k] for k in ("record_id","source","text")):
                raise ValueError("nonempty bound runtime contract required")
            seen.extend(ids)
        if len(seen)!=len(set(seen)) or set(seen)!=set(self.action_ids):
            raise ValueError("each candidate must belong to exactly one explicit slot")

    def specification(self):
        return {**super().specification(),"choice_groups":copy.deepcopy(self.choice_groups)}

    def validate(self):
        super().validate(); self._check_groups()

    @classmethod
    def from_input(cls,sample,choice_groups):
        return cls(workflow=sample.workflow,authority=sample.authority,proposals=sample.proposals,
            contracts=sample.contracts,combinations=sample.combinations,cutoff=sample.cutoff,horizon=sample.horizon,
            event_semantics=sample.event_semantics,choice_groups=choice_groups)


class ExclusiveDispatchWorkflowRiskBN(WorkflowRiskBN):
    MODEL="ExclusiveDispatchWorkflowRiskBN-v1"
    SUPPORTS_EXCLUSIVE_CHOICES=True

    def _choice_parts(self,sample,theta):
        if not isinstance(sample,ExclusiveDispatchInput):
            raise ValueError("explicit runtime-enforced dispatch slots required")
        sample.validate()
        x=super()._design(sample); logits=x@theta
        parts=[]
        for group in sample.choice_groups:
            indices=[sample.action_ids.index(a) for a in group["action_ids"]]
            z=np.r_[0.,logits[indices]]
            logp=z-logsumexp(z)
            parts.append((group,indices,logp))
        return x,parts

    def _occurrence_worlds(self,sample,theta):
        x,parts=self._choice_parts(sample,theta)
        # Keep complete binary indexing for the existing policy CPT exporter.
        # Impossible assignments get exact zero mass, not a soft penalty.
        states=np.asarray(list(product((0.,1.),repeat=len(sample.proposals))))
        log_mass=np.zeros(len(states)); scores=np.zeros((len(states),len(self.theta)))
        valid=np.ones(len(states),dtype=bool)
        for _,indices,logp in parts:
            bits=states[:,indices]; valid&=bits.sum(axis=1)<=1
            selected=bits.argmax(axis=1)+1
            selected[bits.sum(axis=1)==0]=0
            log_mass+=logp[selected]
            scores+=bits@x[indices]-np.exp(logp[1:])@x[indices]
        log_mass[~valid]=-np.inf; scores[~valid]=0.
        return states,log_mass,scores

    def _occurrence_graph_nodes(self,sample,probabilities):
        _,parts=self._choice_parts(sample,self.theta); nodes=[]
        for group,indices,logp in parts:
            for j,i in enumerate(indices):
                earlier=indices[:j]
                # Sequential one-hot binary CPT representation of a categorical
                # choice. Conditional normalizer sums none and remaining options.
                q=float(np.exp(logp[j+1]-logsumexp(np.r_[logp[0],logp[j+1:]])))
                nodes.append({"id":self._event_node_id(sample.action_ids[i]),
                    "parents":[self._event_node_id(sample.action_ids[k]) for k in earlier],
                    "kind":"binary_cpt","probability_one":[0. if any(bits) else q for bits in product((0,1),repeat=len(earlier))],
                    "workflow_node":sample.proposals[i]["node"],"choice_slot":group["id"],
                    "choice_semantics":"none or one actual dispatch; prior alternatives exclude later alternatives",
                    "runtime_evidence":copy.deepcopy(group["evidence"])})
        return nodes

    def predict(self,sample,*,theta=None,derivatives=False,event_conditions=None):
        result=super().predict(sample,theta=theta,derivatives=derivatives,event_conditions=event_conditions)
        result["choice_groups"]=copy.deepcopy(sample.choice_groups)
        result["structural_support"]="At most one dispatch per runtime-enforced slot; none is allowed; impossible multi-option assignments have zero probability."
        result["limitations"]=[
            "Runtime exclusivity must be independently enforced; readable metadata or an agent promise is not proof.",
            "Fixed finite candidate commands known at cutoff, no unlisted products, retries or changing contracts/authority within the forecast window.",
            "Slots are conditionally independent given visible prefix cues; unobserved shared planning outcomes are not yet modeled.",
            "Uses the existing shared-authority and exact scoped-budget fragments. Choice scores do not grant authorization.",
            "Standard conditional softmax; no claim that categorical choices themselves are a novel distribution or a calibrated safety guarantee."]
        return result
