import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from collections import defaultdict, Counter
import spacy
import matplotlib.pyplot as plt
import os

st.title("Product Feature Extraction & Sentiment Grouping (FLAN-T5 Few-Shot)")

def ensure_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_flan_t5():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

@st.cache_resource
def load_spacy():
    return ensure_spacy_model()

tokenizer, model = load_flan_t5()
nlp = load_spacy()

def build_feature_prompt_few_shot(review_text):
    examples = (
        "Example 1:\n"
        "Review: The battery life is excellent.\n"
        "Features:\n- battery life\n\n"
        "Example 2:\n"
        "Review: The phone heats up quickly.\n"
        "Features:\n- phone\n\n"
        "Example 3:\n"
        "Review: Camera quality is disappointing.\n"
        "Features:\n- camera quality\n\n"
        "Example 4:\n"
        "Review: The speaker volume could be louder.\n"
        "Features:\n- speaker volume\n\n"
    )
    return (
        "Extract the main product features mentioned in the following review. "
        "For each feature, include the full phrase as it appears in the review. "
        "List each feature as a separate item.\n\n" +
        examples +
        f"Review: {review_text}\nFeatures:"
    )

def extract_features(text):
    prompt = build_feature_prompt_few_shot(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=1,
            do_sample=False,
            temperature=1.0
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    features = [f.strip("-•, ") for f in decoded.split("\n") if f.strip()]
    if len(features) == 1 and "," in features[0]:
        features = [f.strip() for f in features[0].split(",") if f.strip()]
    return features

def get_sentiment_from_rating(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

def get_core_feature(phrase):
    doc = nlp(phrase)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns[0] if nouns else phrase

uploaded_file = st.file_uploader("Upload your product reviews CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:", df.head())

    review_col = st.selectbox(
        "Select the column containing review text:",
        options=df.columns,
        index=list(df.columns).index("Review Text") if "Review Text" in df.columns else 0
    )

    rating_col = st.selectbox(
        "Select the column for ratings:",
        options=df.columns,
        index=list(df.columns).index("Rating") if "Rating" in df.columns else 0
    )

    if st.button("Extract Features and Group by Sentiment"):
        features_by_sentiment = defaultdict(list)
        sentiments = []
        features_list = []
        core_features_list = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            review = str(row[review_col])
            rating = row[rating_col]
            sentiment = get_sentiment_from_rating(rating)
            sentiments.append(sentiment)
            features = extract_features(review)
            features_list.append(features)
            core_feats = [get_core_feature(f) for f in features]
            core_features_list.append(core_feats)
            features_by_sentiment[sentiment].extend(core_feats)
            progress.progress((i + 1) / len(df))
        df["Extracted_Features"] = features_list
        df["Core_Features"] = core_features_list
        df["Sentiment"] = sentiments
        st.success("Extraction and grouping complete!")

        st.header("Features Grouped by Sentiment")
        for sentiment in ["Positive", "Neutral", "Negative"]:
            feats = features_by_sentiment[sentiment]
            if feats:
                st.subheader(f"{sentiment} Features")
                feat_counts = Counter(feats)
                st.write(pd.DataFrame(feat_counts.most_common(), columns=["Feature", "Count"]))
            else:
                st.write(f"No {sentiment.lower()} features found.")

        all_features = [feature for sublist in df["Core_Features"] for feature in sublist]
        feature_counts = Counter(all_features)
        feature_freq = pd.DataFrame(feature_counts.most_common(), columns=["Feature", "Count"])

        st.subheader("Most Mentioned Product Features")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.barh(feature_freq["Feature"], feature_freq["Count"], color='lightgreen')
        ax2.set_xlabel("Mentions")
        ax2.set_ylabel("Feature")
        ax2.set_title("Most Mentioned Product Features")
        st.pyplot(fig2)

        rating_counts = df[rating_col].value_counts().sort_index()
        st.subheader("Review Ratings Distribution")
        fig1, ax1 = plt.subplots()
        ax1.bar(rating_counts.index.astype(str), rating_counts.values, color='skyblue')
        ax1.set_xlabel("Rating")
        ax1.set_ylabel("Count")
        ax1.set_title("Review Ratings Distribution")
        st.pyplot(fig1)

        st.subheader("Summary Statistics")
        st.markdown(f"""
- **Total unique features extracted:** {feature_freq.shape[0]}
- **Most frequent features:** {', '.join(feature_freq.head(3)['Feature'])}
- **Least frequent features:** {', '.join(feature_freq.tail(7)['Feature'])}
- **Average rating across all reviews:** {df[rating_col].mean():.2f}
""")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="Extracted_Features_Sentiment_Grouped.csv",
            mime="text/csv"
        )
