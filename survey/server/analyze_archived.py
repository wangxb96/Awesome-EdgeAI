#!/usr/bin/env python3
"""Audit recovered server requests without rerunning models or imputing settings."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
import math
import statistics as st

HERE=Path(__file__).resolve().parent
MAIN='1777121713'
FOLLOW='1777295563'
NAMES={'deepseek-r1:14b':'DeepSeek-R1 14B','gemma3:4b':'Gemma3 4B',
       'gpt-oss:20b':'GPT-OSS 20B','llama2:7b':'Llama2 7B','qwen3:8b':'Qwen3 8B'}

def normalized(value):
    return ' '.join(value.strip().split()).lower()

def quantile(values,p):
    values=sorted(values);rank=(len(values)-1)*p;lo=math.floor(rank);hi=math.ceil(rank)
    return values[lo]+(values[hi]-values[lo])*(rank-lo)

def write_csv(name,rows):
    with (HERE/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def summarize(rows,run,model,context='all'):
    # No status/error filtering: every logged row remains in its campaign.
    n=len(rows);lat=[float(r['wall_latency_ms']) for r in rows]
    rate=[float(r['decode_tps']) for r in rows]
    tokens=[int(r['decode_tokens']) for r in rows]
    prompt=[int(r['prompt_tokens']) for r in rows]
    repeats=sorted(set(r['repeat'] for r in rows))
    repeat_means=[st.mean(float(r['wall_latency_ms']) for r in rows if r['repeat']==rep) for rep in repeats]
    energy=[float(r['energy_j']) for r in rows if r.get('energy_j','')!='']
    energy_ratio=sum(energy)/sum(tokens) if len(energy)==n and sum(tokens)>0 else None
    return {'campaign':run,'model':model,'context':context,'n_logged':n,
            'repeat_indices':len(repeats),'logged_errors':sum(bool(r['error']) for r in rows),
            'empty_responses':sum(not r['prediction'].strip() for r in rows),
            'answer_containment':st.mean(int(r['quality_hit']) for r in rows),
            'strict_string_audit':st.mean(not r['error'] and normalized(r['prediction'])==normalized(r['answer']) for r in rows),
            'latency_median_ms':st.median(lat),'latency_p95_linear_ms':quantile(lat,.95),
            'latency_p95_nearest_rank_ms':sorted(lat)[math.ceil(.95*n)-1],
            'latency_mean_ms':st.mean(lat),'latency_request_sd_ms':st.stdev(lat) if n>1 else 0,
            'repeat_latency_mean_sd_ms':st.stdev(repeat_means) if len(repeat_means)>1 else 0,
            'decode_tps_mean':st.mean(rate),'decode_tps_request_sd':st.stdev(rate) if n>1 else 0,
            'endpoint_gpu_energy_proxy_j_per_token':energy_ratio,
            'decoded_min':min(tokens),'decoded_max':max(tokens),
            'decoded_at_128':tokens.count(128),'decoded_at_256':tokens.count(256),
            'prompt_tokens_min':min(prompt),'prompt_tokens_max':max(prompt)}

def main():
    inventory=json.loads((HERE/'archive_inventory.json').read_text())
    for item in inventory['source_code']:
        assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256']
    item=inventory['legacy_plot_aggregate']
    assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['published_sha256']
    ledger=[];all_rows={};energy_verified=0;energy_unverifiable=0
    for item in inventory['files']:
        p=HERE/item['csv'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256']
        run=p.stem.removeprefix('server_suite_raw_')
        rows=list(csv.DictReader(p.open()));all_rows[run]=rows
        assert len({(r['model'],r['context_len'],r['task_id'],r['repeat']) for r in rows})==len(rows)
        assert all(int(r['quality_hit'])==int(not r['error'] and normalized(r['answer']) in normalized(r['prediction'])) for r in rows)
        ledger.append({'campaign':run,'n_logged':len(rows),'models':';'.join(sorted(set(r['model'] for r in rows))),
                       'contexts':';'.join(map(str,sorted(set(int(r['context_len']) for r in rows)))),
                       'repeat_indices':len(set(r['repeat'] for r in rows)),
                       'logged_errors':sum(bool(r['error']) for r in rows),
                       'error_counts':json.dumps(dict(Counter(r['error'] for r in rows if r['error'])),sort_keys=True),
                       'role':'main campaign' if run==MAIN else 'Qwen follow-up' if run==FOLLOW else 'pilot or connectivity probe; not pooled'})
        saved=json.loads((HERE/item['summary']).read_text())
        for q in saved:
            rr=[r for r in rows if r['model']==q['model'] and int(r['context_len'])==int(q['context_len'])]
            if not rr:continue
            ok=[r for r in rr if not r['error']]
            if ok:
                assert math.isclose(st.mean(int(r['quality_hit']) for r in ok),float(q['quality_value']),abs_tol=1e-9)
                assert math.isclose(quantile([float(r['wall_latency_ms']) for r in ok],.95),float(q['p95_latency_ms']),rel_tol=1e-8)
                assert math.isclose(st.mean(float(r['decode_tps']) for r in ok),float(q['tokens_per_sec']),rel_tol=1e-8)
                if q.get('energy_per_token_j') is not None:
                    if all(r.get('energy_j','')!='' for r in rr) and sum(int(r['decode_tokens']) for r in rr)>0:
                        proxy=sum(float(r['energy_j']) for r in rr)/sum(int(r['decode_tokens']) for r in rr)
                        assert math.isclose(proxy,float(q['energy_per_token_j']),rel_tol=1e-8)
                        energy_verified+=1
                    else:
                        energy_unverifiable+=1
    summary=[];cells=[]
    for run in [MAIN,FOLLOW]:
        rows=all_rows[run]
        for model in sorted(set(r['model'] for r in rows)):
            group=[r for r in rows if r['model']==model]
            summary.append(summarize(group,run,model))
            for context in sorted(set(int(r['context_len']) for r in group)):
                cells.append(summarize([r for r in group if int(r['context_len'])==context],run,model,context))
    write_csv('campaign_ledger.csv',ledger);write_csv('reanalysis_by_run.csv',summary);write_csv('reanalysis_by_context.csv',cells)
    tree=ast.parse((HERE/'run_ollama_server_suite.py').read_text())
    tasks=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_TASKS' for t in n.targets))
    answers={task['id']:task['answer'] for task in tasks}
    assert all(r['answer']==answers[r['task_id']] for run in [MAIN,FOLLOW] for r in all_rows[run])
    (HERE/'task_roster.json').write_text(json.dumps({'provenance':'Recovered source-code task definitions; exact historical HTTP payloads were not archived.','tasks':tasks},indent=2)+'\n')
    result={'archive_campaigns':len(ledger),'archive_logged_requests':sum(x['n_logged'] for x in ledger),
            'archive_logged_errors':sum(x['logged_errors'] for x in ledger),
            'main_campaign':MAIN,'main_logged_requests':len(all_rows[MAIN]),'followup_campaign':FOLLOW,'followup_logged_requests':len(all_rows[FOLLOW]),
            'source_scorer_replayed_for_every_logged_row':True,'saved_successful_summary_scores_and_linear_p95_reproduced':True,
            'saved_decode_means_reproduced':True,
            'endpoint_proxy_cells_verified':energy_verified,
            'endpoint_proxy_cells_with_missing_power_fields':energy_unverifiable,
            'primary_canonical_summary_uses_one_campaign':True,'followup_never_pooled_with_main':True,
            'statistical_unit':'Repeated requests on ten fixed tasks; repetition indices are not independently restarted sessions.',
            'confidence_intervals':'Not inferred: independence and population sampling are not established.',
            'energy_boundary':'Endpoint GPU-power proxy, not integrated energy; board IDs and sampler/server alignment unrecorded.',
            'legacy_score_name_corrected':'Answer containment, not strict exact match.',
            'old_plot_conflict':'Legacy plot aggregates use the earlier Qwen run; submitted table uses the later Qwen run. Do not fuse their values.'}
    (HERE/'reanalysis_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
