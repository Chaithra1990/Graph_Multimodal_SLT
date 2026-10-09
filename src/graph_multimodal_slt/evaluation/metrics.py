from __future__ import annotations
import math
from collections import Counter
import sacrebleu
from rouge_score import rouge_scorer

def corpus_bleu4(references:list[str], hypotheses:list[str]):
    # SacreBLEU expects a list of hypothesis strings and one list per reference stream.
    score=sacrebleu.corpus_bleu(hypotheses,[references],tokenize='none',smooth_method='exp')
    return float(score.score), score.get_signature().format()

def corpus_rouge_l(references,hypotheses):
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
    vals=[scorer.score(r,h)['rougeL'].fmeasure for r,h in zip(references,hypotheses)]
    return 100.0*sum(vals)/max(1,len(vals))

def _edit(ref,hyp):
    r=ref.split(); h=hyp.split(); dp=list(range(len(h)+1))
    for i,rt in enumerate(r,1):
        nd=[i]+[0]*len(h)
        for j,ht in enumerate(h,1): nd[j]=min(dp[j]+1,nd[j-1]+1,dp[j-1]+(rt!=ht))
        dp=nd
    return dp[-1],len(r)

def corpus_wer(references,hypotheses):
    edits=words=0
    for r,h in zip(references,hypotheses): e,w=_edit(r,h); edits+=e; words+=w
    return 100.0*edits/max(1,words)

def evaluate_all(references,hypotheses):
    bleu,sig=corpus_bleu4(references,hypotheses); rouge=corpus_rouge_l(references,hypotheses)
    return {'BLEU-4':bleu,'ROUGE-L':rouge,'WER':corpus_wer(references,hypotheses),'sacrebleu_signature':sig}
