"""ChaosNLI (Nie et al. 2020), MNLI subset: every item has 100 independent annotator labels, so the
human label distribution q is known. Uncertainty here is ALEATORIC (annotators disagree); reasoning
cannot remove it. Training never sees q: each visit draws one fresh annotator label Y ~ q.
Split (seed 0): 1000 train / 199 dev / 400 test."""
import random

LABELS = ["entailment", "neutral", "contradiction"]  # dataset label_count order: e, n, c


def load_items(split_seed=0):
    from datasets import load_dataset
    ds = load_dataset("tasksource/chaos-mnli-ambiguity")["train"]
    items = [dict(uid=x["uid"], premise=x["premise"], hypothesis=x["hypothesis"],
                  q=[c / sum(x["label_count"]) for c in x["label_count"]], counts=x["label_count"]) for x in ds]
    random.Random(split_seed).shuffle(items)
    return dict(train=items[:1000], dev=items[1000:1199], test=items[1199:])


def user_message(item):
    return (f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\n\n"
            "Does the premise entail the hypothesis, contradict it, or neither (neutral)? "
            "Think briefly in at most three sentences, then finish with exactly one line of the form "
            "'Answer: entailment', 'Answer: neutral', or 'Answer: contradiction'.")
