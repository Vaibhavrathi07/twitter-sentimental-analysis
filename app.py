import streamlit as st
import pickle
import re

st.set_page_config(page_title='SentiScope', layout='wide')

def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    text = re.sub(r'@\w+', 'USER', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s!?,.-]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text.lower().strip()

@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

with st.sidebar:
    st.title('SentiScope v2')
    st.caption('Improved NLP Model')
    st.divider()
    page = st.radio('Navigate', ['Analyze', 'Metrics', 'History', 'About'])

if 'history' not in st.session_state:
    st.session_state.history = []

if page == 'Analyze':
    st.title('Sentiment Analyzer')
    user_input = st.text_area('Enter a tweet:', placeholder='Type or paste a tweet here...', height=120)
    if st.button('Analyze', type='primary'):
        if user_input.strip():
            cleaned = clean_tweet(user_input)
            pred    = model.predict([cleaned])[0]
            proba   = model.predict_proba([cleaned])[0]
            conf    = max(proba) * 100
            emoji_map = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '🔵', 'Irrelevant': '⚪'}
            st.success(f'{emoji_map.get(pred, "")} **{pred}** — {conf:.1f}% confidence')
            st.progress(int(conf))
            st.session_state.history.insert(0, {
                'tweet': user_input[:60],
                'sentiment': pred,
                'confidence': round(conf, 1)
            })
        else:
            st.warning('Please enter some text first.')

elif page == 'Metrics':
    st.title('Model Performance')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Accuracy', '~87%+')
    c2.metric('Classifier', 'LinearSVC')
    c3.metric('Features', 'Word + Char TF-IDF')
    c4.metric('N-grams', '1-3 word / 3-5 char')

elif page == 'History':
    st.title('Recent Predictions')
    if st.session_state.history:
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button('Clear History'):
            st.session_state.history = []
            st.rerun()
    else:
        st.info('No predictions yet. Go to Analyze tab!')

elif page == 'About':
    st.title('About')
    st.markdown('**Model**: TF-IDF (Word + Char) + LinearSVC')
    st.markdown('**Dataset**: Twitter Entity Sentiment Analysis')
    st.markdown('**Accuracy**: ~87-90% on validation set')
