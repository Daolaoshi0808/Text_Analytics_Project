# text_cleaning_module.py

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin
from tqdm import tqdm
from datasets import load_dataset

# Enable tqdm with pandas
tqdm.pandas()

# Constants for cleaning
STOP = set(stopwords.words("english"))
HEAVY_DROP = {
    "plaintiff", "defendant", "claimant", "respondent", "attorney", "esq", "bar",
    "justice", "judge", "clerk", "motion", "order", "filed", "signed", "entered",
    "docket", "exhibit", "complaint", "appeal", "united", "states", "america",
    "federal", "district", "southern", "northern", "eastern", "western",
    "division", "supreme", "magistrate", "v", "vs", "versus", "agreement",
    "settlement", "section", "chapter", "appendix", "amended", "et", "seq"
}
lemmatizer = WordNetLemmatizer()

date_pattern = (
    r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b"
    r"|\b(?:january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\b"
)
BRACKET_CITE   = r"\[\*+\d+\]"
DOCKET_NO      = r"case no\.[^\n]+"
DISTCT_HEADER  = r"u\.s\.\s+dist\.\s+ct\.[^\n]+"
LEXIS_CITE     = r"pleadings\s+lexis\s+\d+"

def model_clean(text: str) -> str:
    t = text.lower()
    t = re.sub(date_pattern,    " ", t, flags=re.IGNORECASE)
    t = re.sub(BRACKET_CITE,    " ", t)
    t = re.sub(DOCKET_NO,       " ", t)
    t = re.sub(DISTCT_HEADER,   " ", t)
    t = re.sub(LEXIS_CITE,      " ", t)
    t = re.sub(r"in the united states district court.*?(?=\n)", " ", t)
    t = re.sub(r"page\s*\d+\s*of\s*\d+", " ", t)
    t = re.sub(r"http\S+|\S+@\S+|\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = word_tokenize(t)
    return " ".join([
        lemmatizer.lemmatize(w)
        for w in toks if w.isalpha() and w not in STOP and w not in HEAVY_DROP
    ])

class TextCleaner(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna("").progress_apply(self._join_and_clean)

    def _join_and_clean(self, pages):
        if isinstance(pages, list):
            text = " ".join(pages)
        else:
            text = str(pages)
        return model_clean(text)

    def __reduce__(self):
        return (TextCleaner, ())

def extract_case_info(dataset_split):
    df = dataset_split.to_pandas()
    meta_df = pd.json_normalize(df["case_metadata"])
    combined_df = pd.concat([
        df["sources"],
        df["summary/long"],
        df["summary/short"],
        df["summary/tiny"],
        meta_df[["class_action_sought", "case_type"]]
    ], axis=1)
    return combined_df.rename(columns={
        "sources": "full_case",
        "summary/long": "summary_long",
        "summary/short": "summary_short",
        "summary/tiny": "summary_tiny"
    })

def load_clean_filtered_dataset():
    dataset = load_dataset("allenai/multi_lexsum", name="v20230518", trust_remote_code=True)
    train_df = extract_case_info(dataset["train"])
    val_df   = extract_case_info(dataset["validation"])
    test_df  = extract_case_info(dataset["test"])

    # Filter for class action sought: Yes or No
    filter_func = lambda df: df[df["class_action_sought"].isin(["Yes", "No"])]
    return filter_func(train_df), filter_func(val_df), filter_func(test_df)

# Example usage in other file:
# from text_cleaning_module import TextCleaner, load_clean_filtered_dataset
# df_train, df_val, df_test = load_clean_filtered_dataset()
# cleaner = TextCleaner()
# df_train["clean_case"] = cleaner.transform(df_train["full_case"])
