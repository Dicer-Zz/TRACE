import json
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy


# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        n = len(datium)
        if n == 0:
            answers.append("A")
            reasonings.append("")
        else:
            answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
            reasonings.append(datium[2:]) # A/nBecause...
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    evaluation_result = {}
    try:
        bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
        evaluation_result["bleu-1"] = bleu_1

        bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
        evaluation_result["bleu-4"] = bleu_4

        rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
        evaluation_result["rouge-L"] = rouge

        accuracy = caculate_accuracy(outputs["answers"], gts["answers"])
        evaluation_result["accuracy"] = accuracy
    except:
        print("Error in eval_ScienceQA.py")

    # evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result
