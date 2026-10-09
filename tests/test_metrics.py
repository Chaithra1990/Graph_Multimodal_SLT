from graph_multimodal_slt.evaluation.metrics import evaluate_all

def test_metrics_range():
    m=evaluate_all(['a b c'],['a b c'])
    assert 0 <= m['BLEU-4'] <= 100
    assert 0 <= m['ROUGE-L'] <= 100
    assert m['WER'] == 0
