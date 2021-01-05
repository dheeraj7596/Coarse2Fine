import re
import pickle
import pandas as pd


def clean_html(string: str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = string.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = string.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + string)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
        string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
    return string


def clean_email(string: str):
    return " ".join([s for s in string.split() if "@" not in s])


def remove_headers(sent):
    text = []
    subj = None
    data = sent.strip().splitlines()
    for line in data:
        line = line.strip()
        if line.startswith("from:") or line.startswith("lines:") or line.startswith("summary:") or line.startswith(
                "keywords:") or line.startswith("article-i.d.:") or line.startswith("organization:") or line.startswith(
            "nntp-posting-host:") or line.startswith("distribution:") or line.startswith(
            "x-newsreader:") or line.startswith("reply-to:") or line.startswith("news-software:") or line.startswith(
            "originator:"):
            continue
        if line.startswith("subject:"):
            subj = line[len("subject:"):].strip()
        elif len(line) == 0:
            continue
        else:
            text.append(line)
    assert subj is not None
    text = [subj] + text
    return " ".join(text)


def clean_str(string):
    string = remove_headers(string)
    string = clean_html(string)
    string = clean_email(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def clean(df):
    sents = []
    labs = []
    for i, row in df.iterrows():
        # print(row["sentence"])
        # print("*" * 50)
        sent = clean_str(row["sentence"])
        lab = row["label"]
        sents.append(sent)
        labs.append(lab)

    df_clean = pd.DataFrame.from_dict({"text": sents, "label": labs})
    return df_clean


def make_coarse_fine(df, child_to_parent):
    df = df[~df.label.isin(["misc.forsale"])].reset_index(drop=True)
    sents = []
    labs = []
    for i, row in df.iterrows():
        sents.append(row["text"])
        labs.append(child_to_parent[row["label"]])
    df_coarse = pd.DataFrame.from_dict({"text": sents, "label": labs})
    return df, df_coarse


if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df.pkl", "rb"))

    df = clean(df)
    parent_to_child = {
        "computer": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                     "comp.windows.x"],
        "recreation": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
        "science": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
        "politics": ["talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"],
        "religion": ["talk.religion.misc", "alt.atheism", "soc.religion.christian"]
    }
    child_to_parent = {}
    for p in parent_to_child:
        children = parent_to_child[p]
        for ch in children:
            child_to_parent[ch] = p

    df_fine, df_coarse = make_coarse_fine(df, child_to_parent)
    print(df_fine.label.value_counts())
    print(df_coarse.label.value_counts())
    pickle.dump(df_fine, open(base_path + "df_fine.pkl", "wb"))
    pickle.dump(df_coarse, open(base_path + "df_coarse.pkl", "wb"))
    pickle.dump(parent_to_child, open(base_path + "parent_to_child.pkl", "wb"))
