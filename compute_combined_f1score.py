def compute(macro_f_scores, micro_f_scores, num_classes, num_samples):
    """
    :param macro_f_scores: joint macro f1 score of fine-grained classes in a coarse-grained class.
    :param micro_f_scores: joint micro f1 score of fine-grained classes in a coarse-grained class.
    :param num_classes: number of fine-grained classes in this coarse-grained class.
    :param num_samples: number of samples in this coarse-grained class.
    :return:
    """
    micro_f1 = 0
    macro_f1 = 0
    den_macro = 0
    den_micro = 0

    for i, f in enumerate(macro_f_scores):
        macro_f1 += f * num_classes[i]
        micro_f1 += micro_f_scores[i] * num_samples[i]
        den_macro += num_classes[i]
        den_micro += num_samples[i]

    macro_f1 = macro_f1 / den_macro
    micro_f1 = micro_f1 / den_micro
    return macro_f1, micro_f1


if __name__ == "__main__":
    macro_f_scores = []
    micro_f_scores = []
    num_classes = []
    num_samples = []
    macro_f1, micro_f1 = compute(macro_f_scores, micro_f_scores, num_classes, num_samples)
