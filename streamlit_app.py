
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import gensim
from gensim import corpora
from sklearn.preprocessing import LabelEncoder

# --- Constants & Global Variables (must be consistent with training) ---
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

# Label mapping (must be consistent with training)
sentiment_labels_map = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
sentiment_class_names = ['negative', 'neutral', 'positive'] # Ordered as per LabelEncoder fit_transform

topic_name_map = {
    0: "Kecepatan, Proses & Kepuasan",
    1: "Urusan Online & Pembayaran",
    2: "Layanan POLRI & Apresiasi",
    3: "Kendala Pendaftaran & Teknis"
}

# --- Load Assets (Tokenizer, LSTM Model, LDA Model, LabelEncoder) ---
@st.cache_resource
def load_tokenizer_and_model():
    # Load Tokenizer
    with open('tokenizer.json', 'r') as f:
        tokenizer_json_string = f.read()
        loaded_tokenizer = tokenizer_from_json(tokenizer_json_string)

    # Load LSTM Model
    loaded_model = tf.keras.models.load_model('lstm_sentiment_model.h5')

    # Re-initialize LabelEncoder with known classes
    le = LabelEncoder()
    le.fit(sentiment_class_names) # Fit with the exact classes used during training

    return loaded_tokenizer, loaded_model, le

@st.cache_resource
def load_lda_assets():
    # Reconstruct stopwords, stemmer, and normalization_dict for LDA preprocessing
    # This part should ideally be loaded from saved files if they were custom/dynamic
    # For simplicity, re-defining as they were in the notebook
    nltk.download("punkt")

    # =========================
    # STOPWORDS (as defined in xA94dvHYbQzG then 77c3e133)
    # =========================
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'nya', 'dana', 'aplikasi', 'sangat', 'mudah', 'bantu', 'bagus', 'skck']
    stop_words_set = set(stop_factory.get_stop_words())
    stop_words_set.update(more_stopword)

    # =========================
    # STEMMER
    # =========================
    stemmer_obj = StemmerFactory().create_stemmer()

    # =========================
    # NORMALIZATION DICT (as defined in xA94dvHYbQzG)
    # =========================
    normalization_dict_app = {
        'ae': 'saja','aja': 'saja','ajah': 'saja','aj': 'saja','jha': 'saja','sj': 'saja',
        'g': 'tidak','ga': 'tidak','gak': 'tidak','gk': 'tidak','kaga': 'tidak','kagak': 'tidak',
        'kg': 'tidak','ngga': 'tidak','Nggak': 'tidak','tdk': 'tidak','tak': 'tidak',
        'lgi': 'lagi','lg': 'lagi','donlod': 'download','pdhl': 'padahal','pdhal': 'padahal',
        'Coba2': 'coba-coba','tpi': 'tapi','tp': 'tapi','betmanfaat': 'bermanfaat',
        'gliran': 'giliran','kl': 'kalau','klo': 'kalau','gatau': 'tidak tau','bgt': 'banget',
        'hrs': 'harus','dll': 'dan lain-lain','dsb': 'dan sebagainya','trs': 'terus','trus': 'terus',
        'sangan': 'sangat','bs': 'bisa','bsa': 'bisa','gabisa': 'tidak bisa','gbsa': 'tidak bisa',
        'gada': 'tidak ada','gaada': 'tidak ada','gausah': 'tidak usah','bkn': 'bukan',
        'udh': 'sudah','udah': 'sudah','sdh': 'sudah','pertngahn': 'pertengahan',
        'ribet': 'ruwet','ribed': 'ruwet','sdangkan': 'sedangkan','lemot': 'lambat',
        'lag': 'lambat','ngelag': 'gangguan','yg': 'yang','dipakek': 'di pakai','pake': 'pakai',
        'kya': 'seperti','kyk': 'seperti','ngurus': 'mengurus','jls': 'jelas',
        'burik': 'buruk','payah':'buruk','krna': 'karena','dr': 'dari','smpe': 'sampai',
        'slalu': 'selalu','mulu': 'melulu','d': 'di','konek': 'terhubung','suruh': 'disuruh',
        'apk': 'aplikasi','app': 'aplikasi','apps': 'aplikasi','apl': 'aplikasi',
        'bapuk': 'jelek','bukak': 'buka','nyolong': 'mencuri','pas': 'ketika',
        'uodate': 'update','ato': 'atau','onlen': 'online','cmn': 'cuman','jele': 'jelek',
        'angel': 'susah','jg': 'juga','knp': 'kenapa','hbis': 'setelah','tololl': 'tolol','ny': 'nya',
        'skck':'skck','stnk':'stnk','sim':'sim','sp2hp':'sp2hp','propam':'propam','dumas':'dumas',
        'tilang':'tilang','e-tilang':'tilang','etilang':'tilang','surat kehilangan':'kehilangan'
    }

    # Load saved LDA Model and Dictionary
    loaded_lda_model = gensim.models.LdaMulticore.load('lda_model_4_topics.gensim')
    loaded_lda_dictionary = corpora.Dictionary.load('lda_dictionary.gensim')
    
    return stemmer_obj, stop_words_set, normalization_dict_app, loaded_lda_model, loaded_lda_dictionary

tokenizer, model, label_encoder = load_tokenizer_and_model()
stemmer_obj, stop_words_set, normalization_dict_app, lda_model_app, lda_dictionary_app = load_lda_assets()

# --- Preprocessing Functions (Consistent with Notebook) ---
def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\\1{2,}", r"\\1", text)

def preprocess_text(text: str) -> str:
    text = str(text)
    text = normalize_repeated_characters(text)
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_]+:", " ", text)
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r"\\@\\w+|#", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = re.sub(r"[^\\w\\s]+", " ", text)
    text = text.lower()
    for slang, standard in normalization_dict_app.items():
        text = re.sub(rf"\\b{re.escape(slang.lower())}\\b", standard.lower(), text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess_text_lda(text: str) -> str:
    t = stemmer_obj.stem(text)
    tokens = word_tokenize(t)
    tokens = [w for w in tokens if w not in stop_words_set and len(w) > 2]
    return " ".join(tokens)

# Pre-processing function for new sentences for LSTM
def preprocess_new_text_for_lstm(text_input, tok, max_seq_len):
    cleaned_text = preprocess_text(text_input)
    sequence = tok.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_len)
    return padded_sequence, cleaned_text

# Pre-processing function for new sentences for LDA
def preprocess_new_text_for_lda(text):
    cleaned_text = preprocess_text(text)
    cleaned_lda_text = preprocess_text_lda(cleaned_text)
    return cleaned_lda_text

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Sentiment & Topic Analysis for App Reviews")
st.markdown("Enter an app review to get its predicted sentiment and dominant topic.")

user_input = st.text_area("Enter Review Text Here:", "Aplikasi ini sangat bagus dan cepat!")

if st.button("Analyze Review"):
    if user_input:
        with st.spinner("Analyzing..."):
            # --- Sentiment Prediction (LSTM) ---
            padded_text_for_sentiment, cleaned_for_sentiment = preprocess_new_text_for_lstm(user_input, tokenizer, MAX_SEQUENCE_LENGTH)
            
            sentiment_pred_probs = model.predict(padded_text_for_sentiment)
            sentiment_result_encoded = np.argmax(sentiment_pred_probs, axis=1)[0]
            predicted_sentiment = label_encoder.inverse_transform([sentiment_result_encoded])[0]

            # --- Topic Prediction (LDA) ---
            cleaned_lda_text_for_topic = preprocess_new_text_for_lda(user_input)
            bow_for_topic = lda_dictionary_app.doc2bow(cleaned_lda_text_for_topic.split())
            
            if bow_for_topic:
                topic_distribution = lda_model_app.get_document_topics(bow_for_topic)
                if topic_distribution:
                    dominant_topic_id = max(topic_distribution, key=lambda item: item[1])[0]
                    predicted_topic = topic_name_map.get(dominant_topic_id, f"Unknown Topic {dominant_topic_id}")
                else:
                    predicted_topic = "No dominant topic found for this text after LDA processing."
            else:
                predicted_topic = "No relevant words for topic modeling after preprocessing."

            st.subheader("Analysis Results:")
            st.write(f"**Cleaned Review (for LSTM):** {cleaned_for_sentiment}")
            st.write(f"**Predicted Sentiment (LSTM):** {predicted_sentiment}")
            st.write(f"**Predicted Topic (LDA):** {predicted_topic}")

    else:
        st.warning("Please enter some text to analyze.")
