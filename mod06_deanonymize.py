import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_ids = ["age", "gender", "zip3"]

    # count how often combo appears in aux data
    counts = aux_df.groupby(quasi_ids).size().reset_index(name="count")

    # keep combos that appear once
    unique_keys = counts[counts["count"] == 1][quasi_ids]

    # find anonymized records with these combos
    candidates = anon_df.merge(
        unique_keys,
        on=quasi_ids,
        how="inner"
    ) 

    # retrieve corresponding names from aux data
    unique_aux = aux_df.merge(
        unique_keys,
        on=quasi_ids,
        how="inner"
    )

    matches = candidates.merge(
        unique_aux[quasi_ids + ["name"]],
        on=quasi_ids,
        how="inner"
    )

    # return only required columns
    return matches[["anon_id", "name"]].rename(
        columns={"name":"matched_name"}
    ).drop_duplicates()



def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    
    return matches_df["anon_id"].nunique() / len(anon_df)
