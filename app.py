import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="SafeGuard AI - Child Online Safety Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =====================================================
# Model Loading (unchanged logic)
# =====================================================
@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

label_display = {
    "toxic": "Toxic",
    "severe_toxic": "Severely Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}

groupings = {
    "Harassment & Abuse": ["toxic", "severe_toxic", "insult"],
    "Hate Speech": ["identity_hate"],
    "Violence & Threats": ["threat"],
    "Explicit Content": ["obscene"],
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(text: str) -> dict:
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    return {label: float(prob) for label, prob in zip(labels, probs)}


# =====================================================
# Helper UI Functions
# =====================================================
def render_header():
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 3rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem; color: var(--text-color);">SafeGuard AI</h1>
            <p style="font-size: 1.2rem; color: #64748b; max-width: 600px; margin: 0 auto;">
                Real-time detection of harmful content in online messages to help protect children.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_card() -> str:
    st.markdown("#### Enter Message to Analyze")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;'>"
        "Paste chat messages, forum posts, or any text for instant risk assessment.</p>",
        unsafe_allow_html=True,
    )

    placeholder_text = (
        "Example: \"you're such an idiot, go away and never come back\"\n\n"
        "Or try a safe message: \"hey, want to play minecraft later?\""
    )

    user_input = st.text_area(
        label="Message input",
        label_visibility="collapsed",
        placeholder=placeholder_text,
        height=180,
        max_chars=1000,
        key="user_input",
    )

    char_count = len(user_input) if user_input else 0
    st.caption(f"{char_count}/1000 characters")

    return user_input.strip()


def get_verdict(max_prob: float) -> tuple[str, str, str]:
    if max_prob > 0.7:
        return "High Risk", "error", "🛑 High Risk – Likely contains harmful content"
    elif max_prob > 0.4:
        return "Potentially Harmful", "warning", "⚠️ Potentially Harmful – Review recommended"
    else:
        return "Safe", "success", "✅ Safe – No significant harmful content detected"


def render_results(results: dict):
    max_prob = max(results.values())
    highest_label = max(results, key=results.get)

    verdict_title, verdict_type, verdict_message = get_verdict(max_prob)

    # Verdict Card
    st.markdown(f"<h2 style='color: var(--text-color);'>{verdict_title}</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='padding: 1rem; border-radius: 0.75rem; background-color: {'#fee2e2' if verdict_type=='error' else '#fef3c7' if verdict_type=='warning' else '#dcfce7'};'>"
        f"<p style='margin:0; font-size:1.1rem; font-weight:500; color: {'#991b1b' if verdict_type=='error' else '#92400e' if verdict_type=='warning' else '#166534'};'>"
        f"{verdict_message}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Detailed Risk Breakdown")

    # Grouped Progress Bars
    for group_name, group_labels in groupings.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(len(group_labels))
            for idx, label in enumerate(group_labels):
                prob = results[label]
                display_name = label_display[label]

                # Color logic
                if prob > 0.7:
                    color = "red"
                elif prob > 0.4:
                    color = "orange"
                else:
                    color = "green"

                with cols[idx]:
                    st.metric(
                        label=display_name,
                        value=f"{prob:.1%}",
                    )
                    st.progress(prob, text=None)

                    # Highlight highest overall risk
                    if label == highest_label and prob == max_prob:
                        st.caption("← Highest risk category", unsafe_allow_html=False)


def render_transparency_section():
    with st.expander("ℹ️ Model Transparency & Limitations"):
        st.markdown(
            """
            **How it works**  
            This classifier uses a fine-tuned BERT model trained on the Jigsaw Toxic Comment Classification dataset (2018). 
            It predicts the probability of six overlapping toxicity categories independently.

            **Score interpretation**  
            - **>70%**: Strong signal – high confidence in presence  
            - **40–70%**: Moderate signal – warrants review  
            - **<40%**: Low likelihood

            **Limitations**  
            - May produce false positives on sarcasm, reclaimed slurs, or quoted hate speech  
            - Context-dependent nuance (e.g., role-play, education) can be misclassified  
            - Not a replacement for human moderation

            **Dataset**  
            Publicly available from [Jigsaw/Conversation AI](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
            """
        )


def render_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 2rem 0;">
            <p><strong>Disclaimer:</strong> This tool provides probabilistic risk assessment and is not infallible. 
            Always combine with human review for critical decisions.</p>
            <p>Model: BERT-base fine-tuned on Jigsaw Toxic Comments • © 2025 SafeGuard AI Demo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# Main App Flow
# =====================================================
def main():
    render_header()

    user_input = render_input_card()

    analyze_button = st.button(
        "Analyze Message",
        type="primary",
        disabled=(len(user_input) == 0),
        use_container_width=True,
    )

    if analyze_button:
        if user_input:
            with st.spinner("Analyzing message..."):
                results = predict(user_input)

            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            render_results(results)
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            render_transparency_section()
        else:
            st.warning("Please enter some text to analyze.")

    render_footer()


if __name__ == "__main__":
    main()
