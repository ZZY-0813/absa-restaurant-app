"""
Streamlit App for Restaurant Review ABSA (Aspect-Based Sentiment Analysis)
ISOM5240 Deep Learning Course Project

This app provides:
1. Aspect Category Detection (Multi-label Classification)
2. Aspect Sentiment Analysis (4-class Classification)
"""

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json

# ============== Configuration ==============
ASPECT_CATEGORIES = ['food', 'service', 'price', 'ambience', 'anecdotes/miscellaneous']
SENTIMENT_LABELS = ['positive', 'negative', 'neutral', 'conflict']
SENTIMENT_COLORS = {
    'positive': '#90EE90',
    'negative': '#FFB6C1',
    'neutral': '#87CEEB',
    'conflict': '#FFD700'
}

# ============== Model Configuration ==============
# Hugging Face Model URLs
ASPECT_MODEL_NAME = "zhizhi188/results_pipeline1"
SENTIMENT_MODEL_NAME = "zhizhi188/results_pipeline2"

# ============== Model Loading ==============
@st.cache_resource
def load_aspect_detection_model():
    """Load the aspect detection model (Pipeline 1) from Hugging Face Hub"""
    try:
        # Load from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL_NAME)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading aspect detection model: {e}")
        st.info("Please check if the model 'zhizhi188/results_pipeline1' is accessible on Hugging Face Hub.")
        return None, None, None

@st.cache_resource
def load_sentiment_analysis_model():
    """Load the sentiment analysis model (Pipeline 2) from Hugging Face Hub"""
    try:
        # Load from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading sentiment analysis model: {e}")
        st.info("Please check if the model 'zhizhi188/results_pipeline2' is accessible on Hugging Face Hub.")
        return None, None, None

# ============== Prediction Functions ==============
def predict_aspects(text, tokenizer, model, device):
    """
    Pipeline 1: Detect aspect categories mentioned in the review
    Returns: List of detected aspects with probabilities
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Apply threshold
    threshold = 0.5
    predictions = (probs > threshold).astype(int)
    
    detected_aspects = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            detected_aspects.append({
                'aspect': ASPECT_CATEGORIES[i],
                'confidence': float(probs[i])
            })
    
    # Sort by confidence
    detected_aspects.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detected_aspects, probs

def predict_sentiment(text, aspect, tokenizer, model, device):
    """
    Pipeline 2: Predict sentiment for a specific aspect
    Returns: Sentiment label with confidence scores
    """
    # Format input: [ASPECT] aspect [TEXT] text
    combined_text = f"[ASPECT] {aspect} [TEXT] {text}"
    
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_id = np.argmax(probs)
    sentiment = SENTIMENT_LABELS[pred_id]
    confidence = float(probs[pred_id])
    
    # All sentiment probabilities
    all_probs = {
        SENTIMENT_LABELS[i]: float(probs[i])
        for i in range(len(SENTIMENT_LABELS))
    }
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

# ============== Visualization Functions ==============
def create_aspect_distribution_chart(aspects, probs):
    """Create bar chart for aspect detection probabilities"""
    fig = go.Figure()
    
    colors = ['#2ecc71' if p > 0.5 else '#95a5a6' for p in probs]
    
    fig.add_trace(go.Bar(
        x=ASPECT_CATEGORIES,
        y=probs,
        marker_color=colors,
        text=[f"{p:.2f}" for p in probs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Aspect Category Detection Probabilities",
        xaxis_title="Aspect Category",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (0.5)")
    
    return fig

def create_sentiment_pie_chart(sentiment_results):
    """Create pie chart for sentiment distribution"""
    sentiments = [r['sentiment'] for r in sentiment_results]
    sentiment_counts = defaultdict(int)
    for s in sentiments:
        sentiment_counts[s] += 1
    
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = [SENTIMENT_COLORS[s] for s in labels]
    
    fig = px.pie(
        values=values,
        names=labels,
        color=labels,
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution Across Detected Aspects"
    )
    
    fig.update_layout(height=400)
    return fig

def create_sentiment_confidence_chart(sentiment_results):
    """Create bar chart showing sentiment confidence for each aspect"""
    aspects = [r['aspect'] for r in sentiment_results]
    confidences = [r['confidence'] for r in sentiment_results]
    sentiments = [r['sentiment'] for r in sentiment_results]
    colors = [SENTIMENT_COLORS[s] for s in sentiments]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=aspects,
        y=confidences,
        marker_color=colors,
        text=[f"{s}\n{c:.2f}" for s, c in zip(sentiments, confidences)],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Sentiment Confidence by Aspect",
        xaxis_title="Aspect",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    return fig

# ============== Streamlit UI ==============
def main():
    # Page configuration
    st.set_page_config(
        page_title="Restaurant Review ABSA System",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🍽️ Restaurant Review Aspect-Based Sentiment Analysis")
    st.markdown("""
    **ISOM5240 Deep Learning Course Project**
    
    This application analyzes restaurant reviews to identify:
    - **Aspect Categories**: Food, Service, Price, Ambience, Anecdotes/Miscellaneous
    - **Sentiment**: Positive, Negative, Neutral, Conflict
    
    Powered by Hugging Face Transformers (DistilBERT & RoBERTa)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Model Architecture:**
        - Pipeline 1: Multi-label Aspect Detection
        - Pipeline 2: 4-class Sentiment Analysis
        
        **Base Models:**
        - DistilBERT (Aspect Detection)
        - RoBERTa (Sentiment Analysis)
        
        **Dataset:**
        - SemEval-2014 Task 4
        - 3,044 training samples
        - 100 validation samples
        """)
        
        st.markdown("---")
        st.markdown("**Developed by:** 梁智童 (Zhitong Liang)")
        st.markdown("**GitHub:** [https://github.com/ZZY-0813/absa-restaurant-app.git]")
        st.markdown("**HuggingFace:** [zhizhi188/results_pipeline1 zhizhi188/results_pipeline2]")
    
    # Load models
    with st.spinner("Loading models..."):
        aspect_tokenizer, aspect_model, aspect_device = load_aspect_detection_model()
        sentiment_tokenizer, sentiment_model, sentiment_device = load_sentiment_analysis_model()
    
    if aspect_model is None or sentiment_model is None:
        st.error("Failed to load models. Please check the model paths.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Input section
    st.header("📝 Enter Restaurant Review")
    
    # Example reviews
    example_reviews = [
        "The food was absolutely delicious but the service was slow and the prices were too high.",
        "Great atmosphere with beautiful decor, and the staff was very friendly and attentive!",
        "The pasta was overcooked and the waiter was rude. However, the dessert was amazing.",
        "Reasonable prices for good quality food. The ambience was nice but nothing special."
    ]
    
    selected_example = st.selectbox(
        "Or select an example review:",
        ["Custom..."] + example_reviews
    )
    
    if selected_example == "Custom...":
        review_text = st.text_area(
            "Type your review here:",
            height=100,
            placeholder="Enter your restaurant review..."
        )
    else:
        review_text = selected_example
        st.text_area("Review:", review_text, height=100, disabled=True)
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    if analyze_button and review_text:
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with st.spinner("Analyzing..."):
            # ========== Pipeline 1: Aspect Detection ==========
            detected_aspects, aspect_probs = predict_aspects(
                review_text, aspect_tokenizer, aspect_model, aspect_device
            )
            
            with col1:
                st.subheader("📊 Pipeline 1: Aspect Detection")
                
                if detected_aspects:
                    st.success(f"Detected {len(detected_aspects)} aspect(s)")
                    
                    # Display detected aspects
                    for aspect in detected_aspects:
                        st.markdown(f"- **{aspect['aspect'].capitalize()}** (confidence: {aspect['confidence']:.3f})")
                else:
                    st.warning("No aspects detected in this review.")
                
                # Aspect probability chart
                fig_aspect = create_aspect_distribution_chart(detected_aspects, aspect_probs)
                st.plotly_chart(fig_aspect, use_container_width=True)
            
            # ========== Pipeline 2: Sentiment Analysis ==========
            with col2:
                st.subheader("💭 Pipeline 2: Sentiment Analysis")
                
                if detected_aspects:
                    sentiment_results = []
                    
                    for aspect_info in detected_aspects:
                        aspect = aspect_info['aspect']
                        result = predict_sentiment(
                            review_text, aspect, 
                            sentiment_tokenizer, sentiment_model, sentiment_device
                        )
                        result['aspect'] = aspect
                        sentiment_results.append(result)
                        
                        # Display sentiment for each aspect
                        emoji = {
                            'positive': '😊',
                            'negative': '😞',
                            'neutral': '😐',
                            'conflict': '😵'
                        }
                        
                        sentiment_color = SENTIMENT_COLORS[result['sentiment']]
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 5px solid {sentiment_color}; margin: 5px 0;">
                            <b>{aspect.capitalize()}</b>: {emoji[result['sentiment']]} {result['sentiment'].capitalize()} 
                            (confidence: {result['confidence']:.3f})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sentiment visualization
                    st.markdown("---")
                    
                    # Sentiment distribution pie chart
                    fig_sentiment_pie = create_sentiment_pie_chart(sentiment_results)
                    st.plotly_chart(fig_sentiment_pie, use_container_width=True)
                    
                    # Sentiment confidence chart
                    fig_sentiment_conf = create_sentiment_confidence_chart(sentiment_results)
                    st.plotly_chart(fig_sentiment_conf, use_container_width=True)
                else:
                    st.info("No aspects detected, skipping sentiment analysis.")
        
        # ========== Summary Section ==========
        st.markdown("---")
        st.header("📋 Analysis Summary")
        
        if detected_aspects:
            # Create summary table
            summary_data = []
            for sr in sentiment_results:
                summary_data.append({
                    'Aspect': sr['aspect'].capitalize(),
                    'Sentiment': sr['sentiment'].capitalize(),
                    'Confidence': f"{sr['confidence']:.3f}",
                    'Positive%': f"{sr['all_probabilities']['positive']:.1%}",
                    'Negative%': f"{sr['all_probabilities']['negative']:.1%}",
                    'Neutral%': f"{sr['all_probabilities']['neutral']:.1%}",
                    'Conflict%': f"{sr['all_probabilities']['conflict']:.1%}"
                })
            
            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Overall sentiment
            st.markdown("### Overall Review Sentiment")
            
            # Calculate weighted sentiment
            positive_count = sum(1 for r in sentiment_results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in sentiment_results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in sentiment_results if r['sentiment'] == 'neutral')
            conflict_count = sum(1 for r in sentiment_results if r['sentiment'] == 'conflict')
            
            total = len(sentiment_results)
            
            col_pos, col_neg, col_neu, col_con = st.columns(4)
            
            with col_pos:
                st.metric("Positive", f"{positive_count} ({positive_count/total:.1%})")
            with col_neg:
                st.metric("Negative", f"{negative_count} ({negative_count/total:.1%})")
            with col_neu:
                st.metric("Neutral", f"{neutral_count} ({neutral_count/total:.1%})")
            with col_con:
                st.metric("Conflict", f"{conflict_count} ({conflict_count/total:.1%})")
        else:
            st.info("No aspects detected in this review. Try entering a more detailed review.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>ISOM5240 Deep Learning | Aspect-Based Sentiment Analysis Project</p>
        <p>Powered by <a href="https://huggingface.co">Hugging Face</a> & <a href="https://streamlit.io">Streamlit</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
