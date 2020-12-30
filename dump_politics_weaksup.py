import pickle
from sklearn.metrics import classification_report


def measure_quality(df_fine, temp_df, l):
    true = []
    preds = [l] * len(temp_df)
    for i, row in temp_df.iterrows():
        inds = df_fine[df_fine["text"] == row["text"]].index.values
        true.append(df_fine["label"][inds[0]])
    print(classification_report(true, preds), flush=True)


def func(df_fine, df, p="politics"):
    parent_df = df[df["label"].isin([p])].reset_index(drop=True)

    print("surveillance")
    reg_exp = "|".join(
        ["gun control", "law enforcement", "immigration", "abortion", "gay rights", "federal budget",
         "the affordable care act", "affordable care act", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("surveillance") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "surveillance")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "surveillance" + ".pkl", "wb"))

    print("gun_control")
    reg_exp = "|".join(
        ["surveillance", "law enforcement", "immigration", "abortion", "gay rights", "federal budget",
         "the affordable care act", "affordable care act", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("gun control") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "gun_control")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "gun_control" + ".pkl", "wb"))

    print("law_enforcement")
    reg_exp = "|".join(
        ["surveillance", "gun control", "immigration", "gay", "abortion", "gay rights",
         "federal budget", "budget", "the affordable care act", "affordable care act", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("law enforcement") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "law_enforcement")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "law_enforcement" + ".pkl", "wb"))

    print("immigration")
    reg_exp = "|".join(
        ["surveillance", "gun control", "law enforcement",
         "abortion", "gay rights", "gay", "federal budget", "budget", "the affordable care act",
         "affordable care act", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("immigration") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "immigration")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "immigration" + ".pkl", "wb"))

    print("abortion")
    reg_exp = "|".join(
        ["surveillance", "gun control", "gun", "law enforcement",
         "immigration", "gay rights", "gay", "federal budget", "budget", "the affordable care act",
         "affordable care act", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("abortion") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "abortion")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "abortion" + ".pkl", "wb"))

    print("gay_rights")
    reg_exp = "|".join(
        ["surveillance", "gun control", "law enforcement", "immigration",
         "abortion", "federal budget", "budget", "the affordable care act", "affordable care act",
         "military"])
    child_df = parent_df[
        parent_df.text.str.contains("gay rights") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "gay_rights")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "gay_rights" + ".pkl", "wb"))

    print("federal_budget")
    reg_exp = "|".join(
        ["surveillance", "gun control", "law enforcement", "immigration",
         "abortion", "gay rights", "gay", "the affordable care act", "affordable care act", "insurance", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("budget") & ~parent_df.text.str.contains(
            reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "federal_budget")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "federal_budget" + ".pkl", "wb"))

    print("the_affordable_care_act")
    reg_exp = "|".join(
        ["surveillance", "gun control", "law enforcement", "immigration", "abortion", "gay rights", "federal budget",
         "budget", "military"])
    child_df = parent_df[
        parent_df.text.str.contains("the affordable care act") & ~parent_df.text.str.contains(reg_exp)].reset_index(
        drop=True)
    measure_quality(df_fine, child_df, "the_affordable_care_act")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "the_affordable_care_act" + ".pkl", "wb"))

    print("military")
    reg_exp = "|".join(
        ["surveillance", "gun control", "law", "enforcement", "law enforcement", "immigration",
         "abortion", "gay rights", "gay", "federal budget", "budget", "the affordable care act",
         "affordable care act", "insurance"])
    child_df = parent_df[
        parent_df.text.str.contains("military") & ~parent_df.text.str.contains(reg_exp)].reset_index(drop=True)
    measure_quality(df_fine, child_df, "military")
    pickle.dump(child_df, open(pkl_dump_dir + "exclusive/" + "military" + ".pkl", "wb"))


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    df_fine = pickle.load(open(pkl_dump_dir + "df_fine.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))
    func(df_fine, df)
