"""Unsupervised clustering of task names across every schedule.

The rule files answer "which trade / phase / stage is this?". They cannot
answer "what kinds of work actually recur across these eleven programmes?",
because nobody wrote a rule for work nobody anticipated. So this module lets
the task names group themselves:

    clean text -> TF-IDF (words + character n-grams) -> LSA -> k-means

One global clustering is fitted over all schedules at once, so cluster 7 means
the same thing in DC30 as it does in USB and the clusters are comparable
between projects.

Two deliberate choices:

- *Fitted on unique cleaned names, not on rows.* "Ausführung Maler" appears
  hundreds of times; fitting on rows would let a handful of boilerplate
  phrases drag the centroids around. Every row is assigned afterwards.
- *Character n-grams alongside words.* German compounds mean "Kabeltrasse",
  "Kabeltrassen" and "ELT-Trassen" share no word token but plenty of
  character 4-grams.

The clusters are then used to fill in the montage stage. The keyword rules in
`montage_rules.yaml` only reach ~62% of work tasks; if a cluster is clearly a
Grobmontage cluster where the rules did fire, the unlabelled members of that
same cluster are very probably Grobmontage too. That inference is recorded as
`stage_source == "cluster"` so it never passes for direct evidence.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .classify import normalize_text

# 80 rather than the ~40 that reads more comfortably as a bar chart: at 40 the
# generic installation verbs ("montage", "einbau", "installation") collapse into
# one 4'300-task catch-all. Going to 80 halves the largest cluster's share of
# rows (15.2% -> 10.7%) and lifts the silhouette from 0.14 to 0.22, and the
# charts show the top N clusters anyway.
N_CLUSTERS = 80
RANDOM_STATE = 42

# Structural noise in task names. Without this the pRED schedules cluster by
# their "B04-Dach-B.03-BAU_" prefix and everything else clusters by floor.
NOISE_PATTERNS = [
    r"\bb\d{2}-\S*?_",                       # pRED path prefix B05-EG-B.24-BAU_
    r"\(\s*ms\s*\d+[^)]*\)",                 # (MS1), (MS2 - MS4)
    r"\bms\d+\b",
    r"\bsia\s*\d+(?:\.\d+)*",                # SIA 312.1
    r"\b(?:ober|unter)?geschoss\s*\d+\b",
    r"\b\d+\s*\.\s*(?:og|ug)\b",             # 6.OG
    r"\b(?:og|ug)\s*\d+\b",
    r"\b\d+(?:og|ug)\b",
    r"\bbkp\s*\d+(?:\.\d+)*",
    r"\b\d+(?:\.\d+)+\b",                    # dotted codes 312.4
    r"\bwoche\s*\d+\b",
    r"\bkw\s*\d+\b",
]

# Pure function words. Kept short on purpose: words like "start", "ende" or
# "dach" look like filler but carry real meaning in a construction schedule.
STOPWORDS = {
    "und", "oder", "der", "die", "das", "des", "den", "dem", "ein", "eine",
    "einer", "eines", "fuer", "von", "vom", "im", "in", "an", "auf", "aus",
    "mit", "ohne", "bis", "ab", "zum", "zur", "zu", "am", "als", "bei",
    "sowie", "inkl", "etc", "div", "diverse", "alle", "aller", "allen",
    "je", "pro", "nach", "vor", "ueber", "unter", "the", "of", "and", "for",
}


def clean_name(value: object) -> str:
    """Strip codes, floors and numbering so only the work description is left."""
    text = normalize_text(value)
    if not text:
        return ""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z]+", " ", text)          # drops digits and punctuation
    tokens = [
        token
        for token in text.split()
        if len(token) > 2 and token not in STOPWORDS
    ]
    return " ".join(tokens)


class _Space:
    """The fitted text space, so new names can be placed in the same geometry."""

    def __init__(self, texts: list[str]) -> None:
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 5), min_df=3, sublinear_tf=True
        )
        self.words = self.word_vec.fit_transform(texts)
        chars = self.char_vec.fit_transform(texts)
        combined = self._combine(self.words, chars)

        components = int(min(150, combined.shape[1] - 1, len(texts) - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
        self.reduced = normalize(self.svd.fit_transform(combined))

    @staticmethod
    def _combine(words, chars):
        # Half the weight on characters: they are far more numerous than word
        # features, so unweighted they would drown the words out.
        return hstack([words, chars * 0.5]).tocsr()

    def transform(self, texts: list[str]) -> np.ndarray:
        combined = self._combine(
            self.word_vec.transform(texts), self.char_vec.transform(texts)
        )
        return normalize(self.svd.transform(combined))


def _cluster_labels(
    words: np.ndarray,
    vectorizer: TfidfVectorizer,
    assignments: np.ndarray,
    n_clusters: int,
    terms_per_label: int = 3,
) -> dict[int, str]:
    """Name each cluster after its strongest word features."""
    vocabulary = np.array(vectorizer.get_feature_names_out())
    labels: dict[int, str] = {}
    for cluster in range(n_clusters):
        rows = words[assignments == cluster]
        if rows.shape[0] == 0:
            labels[cluster] = f"Cluster {cluster}"
            continue
        weights = np.asarray(rows.mean(axis=0)).ravel()
        chosen: list[str] = []
        for index in np.argsort(weights)[::-1]:
            if weights[index] <= 0:
                break
            term = vocabulary[index]
            # Skip terms already implied by one we picked ("montage" after
            # "montage kabeltrassen") so the label carries three real ideas.
            if any(term in picked or picked in term for picked in chosen):
                continue
            chosen.append(term)
            if len(chosen) == terms_per_label:
                break
        labels[cluster] = " / ".join(chosen) if chosen else f"Cluster {cluster}"
    return labels


def assign_clusters(frame: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Add `cluster_id` / `cluster_label` to every named task.

    Fitted on the unique cleaned names of work tasks, then applied to all
    named rows including summaries, so the exported file covers everything.
    """
    out = frame.copy()
    out["clean_name"] = out["task_name"].map(clean_name)

    named = out["clean_name"].str.len() > 0
    fit_texts = sorted(set(out.loc[named & out["is_executable"], "clean_name"]))
    if len(fit_texts) < n_clusters * 2:
        print(f"  only {len(fit_texts)} distinct work names, clustering skipped")
        out["cluster_id"] = pd.NA
        out["cluster_label"] = pd.NA
        return out

    space = _Space(fit_texts)
    model = KMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10
    ).fit(space.reduced)

    labels = _cluster_labels(space.words, space.word_vec, model.labels_, n_clusters)
    text_to_cluster = dict(zip(fit_texts, model.labels_))

    # Rows whose name never entered the fit (summaries, milestones, names seen
    # only outside work tasks) go to the nearest centroid of the same space.
    unseen = sorted(set(out.loc[named, "clean_name"]) - set(text_to_cluster))
    if unseen:
        predicted = model.predict(space.transform(unseen))
        text_to_cluster.update(dict(zip(unseen, predicted)))

    out["cluster_id"] = out["clean_name"].map(text_to_cluster).astype("Int64")
    out["cluster_label"] = out["cluster_id"].map(labels).astype("string")

    sample = min(4000, len(space.reduced))
    rng = np.random.default_rng(RANDOM_STATE)
    index = rng.choice(len(space.reduced), size=sample, replace=False)
    score = silhouette_score(space.reduced[index], model.labels_[index])
    print(
        f"  {n_clusters} clusters over {len(fit_texts)} distinct work names, "
        f"silhouette {score:.3f}"
    )
    return out.drop(columns=["clean_name"])


# ------------------------------------------------------------ stage inference

MIN_STAGED_TASKS = 15
MIN_STAGED_SHARE = 0.30
MIN_DOMINANT_SHARE = 0.50

STAGE_ORDER = {"GROB": 10, "FEIN": 20, "END": 30}
STAGE_NAMES = {"GROB": "Grobmontage", "FEIN": "Feinmontage", "END": "Endmontage"}


def cluster_stage_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: how much of it the rules staged, and which stage dominates."""
    work = frame[frame["is_executable"] & frame["cluster_id"].notna()]
    if work.empty:
        return pd.DataFrame()

    rows = []
    for cluster_id, group in work.groupby("cluster_id", sort=True):
        staged = group[group["montage_code"].notna()]
        row = {
            "cluster_id": int(cluster_id),
            "cluster_label": group["cluster_label"].iloc[0],
            "tasks": int(len(group)),
            "staged_tasks": int(len(staged)),
            "staged_share": float(len(staged) / len(group)),
            "dominant_stage": pd.NA,
            "dominant_share": float("nan"),
            "inferable": False,
        }
        if not staged.empty:
            by_stage = staged.groupby("montage_code")["duration_days"].sum()
            if by_stage.sum() > 0:
                shares = by_stage / by_stage.sum()
                row["dominant_stage"] = str(shares.idxmax())
                row["dominant_share"] = float(shares.max())
        row["inferable"] = bool(
            row["staged_tasks"] >= MIN_STAGED_TASKS
            and row["staged_share"] >= MIN_STAGED_SHARE
            and row["dominant_share"] >= MIN_DOMINANT_SHARE
        )
        rows.append(row)

    profile = pd.DataFrame(rows)
    top_gewerk = {
        int(cluster_id): (
            group["gewerk_name"].mode().iloc[0]
            if not group["gewerk_name"].mode().empty
            else pd.NA
        )
        for cluster_id, group in work.groupby("cluster_id", sort=True)
    }
    profile["top_gewerk"] = profile["cluster_id"].map(top_gewerk)
    return profile


def apply_stage_inference(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill the montage stage from each cluster's dominant stage.

    `montage_code` is left untouched as the rule-based record. The effective
    stage used by the charts lands in `stage_code`, with `stage_source`
    saying whether it came from a parsed field, a keyword rule or a cluster.
    """
    out = frame.copy()
    out["stage_code"] = out["montage_code"]
    out["stage_source"] = out["montage_source"]

    profile = cluster_stage_profile(out)
    if not profile.empty:
        inferable = profile[profile["inferable"]]
        stage_by_cluster = dict(
            zip(inferable["cluster_id"], inferable["dominant_stage"])
        )
        fillable = (
            out["stage_code"].isna()
            & out["cluster_id"].notna()
            & out["is_executable"]
        )
        inferred = out.loc[fillable, "cluster_id"].map(stage_by_cluster).dropna()
        out.loc[inferred.index, "stage_code"] = inferred
        out.loc[inferred.index, "stage_source"] = "cluster"

    out["stage_name"] = out["stage_code"].map(STAGE_NAMES).astype("string")
    out["stage_order"] = out["stage_code"].map(STAGE_ORDER).astype("Int64")
    return out, profile
