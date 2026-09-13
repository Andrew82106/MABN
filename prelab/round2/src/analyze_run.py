"""Sequential analysis of an already finished generation run."""
import argparse
import subprocess
import sys
from common import ROOT,readl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--task',required=True); args=ap.parse_args()
    run=ROOT/f'data/{args.task}_{args.model}'
    expected={'news':394,'trivia':2700,'rag':1050}[args.task]
    assert (run/'manifest.json').exists() and len(readl(run/'outputs.jsonl'))==expected,'Generation is not complete'
    commands=[['label_outputs.py'],['probes.py']]
    if args.task=='rag' and args.model=='large': commands.insert(0,['fix_generation_tokens.py'])
    if args.task!='trivia': commands.append(['probes.py','--supervised','--workers','3'])
    if args.task=='news': commands.append(['limited_labels.py'])
    commands.append(['evaluate.py'])
    if args.task=='trivia' and args.model=='large': commands.append(['reviewed_sample_metrics.py'])
    for script,*flags in commands:
        name=script.replace('.py','')+('_supervised' if flags else '')
        log=ROOT/f'logs/{name}_{args.task}_{args.model}.log'
        print('RUN',name,args.task,args.model,flush=True)
        with log.open('w',encoding='utf8') as f:
            task_args=[] if script=='limited_labels.py' else ['--task',args.task]
            subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'src'/script),*task_args,'--model',args.model,*flags],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('ANALYSIS COMPLETE',args.task,args.model,flush=True)

if __name__=='__main__': main()
