import pickle
import pandas as pd


def create_parent_to_child(base_path):
    parent_to_child = {}
    f = open(base_path + "label_hier.txt", "r")
    hier = f.readlines()
    f.close()
    hier = hier[1:]
    for line in hier:
        tokens = line.strip().split()
        parent_to_child[tokens[0]] = tokens[1:]
    return parent_to_child


if __name__ == "__main__":
    base_path = "./"
    f = open(base_path + "dataset.txt", "r")
    data_lines = f.readlines()
    f.close()

    f = open(base_path + "labels.txt", "r")
    label_lines = f.readlines()
    f.close()

    assert len(data_lines) == len(label_lines)
    length = len(data_lines)

    parent_to_child = create_parent_to_child(base_path)
    parent_to_child["cs"].remove("cs.OH")
    parent_to_child["physics"].remove("physics.gen-ph")
    parent_to_child["physics"].remove("physics.class-ph")
    parent_to_child["math"].remove("math.GM")
    child_to_parent = {}
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            child_to_parent[ch] = p

    sents = []
    coarse_labels = []
    fine_labels = []

    for i in range(length):
        sent = data_lines[i].lower().strip()
        fine_lbl = label_lines[i].strip()
        if fine_lbl in ["cs.OH", "physics.gen-ph", "physics.class-ph", "math.GM"]:
            continue
        coarse_lbl = child_to_parent[fine_lbl]
        sents.append(sent)
        coarse_labels.append(coarse_lbl)
        fine_labels.append(fine_lbl)

    df_coarse = pd.DataFrame.from_dict({"text": sents, "label": coarse_labels})
    df_fine = pd.DataFrame.from_dict({"text": sents, "label": fine_labels})
    print(df_coarse.label.value_counts())
    print(df_fine.label.value_counts())
    for p in parent_to_child:
        print(p, len(parent_to_child[p]))
    # pickle.dump(df_coarse, open(base_path + "df_coarse.pkl", "wb"))
    # pickle.dump(df_fine, open(base_path + "df_fine.pkl", "wb"))
    # pickle.dump(parent_to_child, open(base_path + "parent_to_child.pkl", "wb"))
