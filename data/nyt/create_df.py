import pickle
from pandas import DataFrame

if __name__ == "__main__":
    base_path = "./"
    f = open(base_path + "dataset.txt", "r")
    lines = f.readlines()
    f.close()
    f = open(base_path + "labels.txt", "r")
    labels = f.readlines()
    f.close()

    parent_labels = ["politics", "arts", "business", "science", "sports"]
    child_labels = ["federal_budget", "surveillance", "the_affordable_care_act", "immigration", "law_enforcement",
                    "gay_rights", "gun_control", "military", "abortion", "dance", "television", "music", "movies",
                    "stocks_and_bonds", "energy_companies", "economy", "international_business", "cosmos",
                    "environment", "hockey", "basketball", "tennis", "golf", "football", "baseball", "soccer"]

    child_to_parent = {"federal_budget": "politics",
                       "surveillance": "politics",
                       "the_affordable_care_act": "politics",
                       "immigration": "politics",
                       "law_enforcement": "politics",
                       "gay_rights": "politics",
                       "gun_control": "politics",
                       "military": "politics",
                       "abortion": "politics",
                       "dance": "arts",
                       "television": "arts",
                       "music": "arts",
                       "movies": "arts",
                       "stocks_and_bonds": "business",
                       "energy_companies": "business",
                       "economy": "business",
                       "international_business": "business",
                       "cosmos": "science",
                       "environment": "science",
                       "hockey": "sports",
                       "basketball": "sports",
                       "tennis": "sports",
                       "golf": "sports",
                       "football": "sports",
                       "baseball": "sports",
                       "soccer": "sports"
                       }

    parent_to_child = {}
    for child in child_to_parent:
        try:
            parent_to_child[child_to_parent[child]].append(child)
        except:
            parent_to_child[child_to_parent[child]] = [child]

    dic = {"text": [], "label": []}
    for i, line in enumerate(lines):
        label = labels[i].strip()
        sent = lines[i].strip().lower()
        dic["text"].append(sent)
        dic["label"].append(label)

    df = DataFrame.from_dict(dic)

    assert len(set(df.label) - set(parent_labels) - set(child_labels)) == 0

    df_fine = df[df.label.isin(child_labels)]

    dic = {"text": [], "label": []}
    for i, row in df_fine.iterrows():
        label = row["label"]
        dic["text"].append(row["text"])
        dic["label"].append(child_to_parent[label])

    df_coarse = DataFrame.from_dict(dic)

    print("Length of df, df_coarse, df_fine", len(df), len(df_coarse), len(df_fine))
    pickle.dump(df, open(base_path + "df.pkl", "wb"))
    pickle.dump(df_coarse, open(base_path + "df_coarse.pkl", "wb"))
    pickle.dump(df_fine, open(base_path + "df_fine.pkl", "wb"))
    pickle.dump(child_to_parent, open(base_path + "child_to_parent.pkl", "wb"))
    pickle.dump(parent_to_child, open(base_path + "parent_to_child.pkl", "wb"))
