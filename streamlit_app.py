
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
    stemmer_factory = StemmerFactory()
    stemmer_obj = stemmer_factory.create_stemmer()

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

    # Load LDA Model (assuming lda_model_4_topics was the final one)
    # This part requires re-creating the dictionary and corpus used for lda_model_4_topics
    # We need access to the original 'texts' and dictionary filtering parameters
    # For a real deployment, save the LDA model and dictionary explicitly
    
    # To approximate: create a dummy dictionary and corpus with the same filtering applied
    # For accurate reproduction, the dictionary and model should be saved and loaded.
    # As an alternative, we will re-generate a 'mock' dictionary using pre-loaded texts. 
    # In a real scenario, you would save and load dictionary.gensim and lda_model.gensim

    # We don't have df_lda here, so we need a placeholder to reconstruct the dictionary. 
    # This is a limitation if the full dictionary/corpus isn't saved.
    # For a proper deployment, the dictionary and lda_model itself should be saved (e.g., using gensim.corpora.Dictionary.save and gensim.models.LdaMulticore.save)
    st.warning("LDA model and dictionary not explicitly saved/loaded. Topic prediction will be based on a re-initialized dictionary which might differ slightly if not built from original texts.")

    # Minimalistic approach: just load the lda model if it was saved as a whole
    # Since the notebook saved it to a variable 'lda_model_4_topics' but not to disk,
    # we need to simulate its creation. This part would be `gensim.models.LdaMulticore.load('lda_model_4_topics.gensim')`
    # and `gensim.corpora.Dictionary.load('dictionary.gensim')` in a real app.
    # For now, let's assume `lda_model_4_topics` and `dictionary` are available from the colab env for this `%%writefile` context if run directly after training
    
    # --- Re-using global variables from Colab environment if they exist ---
    # In a standalone Streamlit app, you would load these from saved files.
    # For example: lda_model = gensim.models.LdaMulticore.load('lda_model_4_topics')
    #               dictionary = corpora.Dictionary.load('lda_dictionary')

    # --- For this demo, let's just make sure required globals are passed/recreated ---
    # This assumes `lda_model_4_topics` and `dictionary` are accessible after `%%writefile` is executed
    # If running this as a separate file, you MUST save and load the LDA model and dictionary.
    return stemmer_obj, stop_words_set, normalization_dict_app

tokenizer, model, label_encoder = load_tokenizer_and_model()
stemmer_obj, stop_words_set, normalization_dict_app = load_lda_assets()

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
            # For a proper Streamlit app, `lda_model_4_topics` and `dictionary` need to be loaded.
            # Since they are not saved to disk in the notebook, we will use a simplified approach.
            # For this Streamlit app to work stand-alone, you MUST save your LDA model and dictionary. 
            # Example: lda_model_4_topics.save('lda_model_4_topics.gensim') 
            #          dictionary.save('lda_dictionary.gensim')
            # Then load them here:
            # loaded_lda_model = gensim.models.LdaMulticore.load('lda_model_4_topics.gensim')
            # loaded_lda_dictionary = corpora.Dictionary.load('lda_dictionary.gensim')

            # For now, we'll try to reconstruct the dictionary from original df_lda if possible, or give a warning.
            # THIS PART IS A SIMPLIFICATION. For robust deployment, save/load LDA assets.
            try:
                # This requires 'df_lda' and its 'cleaned_content_lda' to be available which is not typical for a standalone script
                # Recreating from scratch based on pre-processing functions here
                texts_for_dict = [preprocess_text_lda(r) for r in df['cleaned_content']] # Assuming df is available or loaded
                temp_dictionary = corpora.Dictionary([t.split() for t in texts_for_dict])
                temp_dictionary.filter_extremes(no_below=30, no_above=0.7) # Consistent with 4-topic model
                
                cleaned_lda_text_for_topic = preprocess_text_lda(user_input)
                bow_for_topic = temp_dictionary.doc2bow(cleaned_lda_text_for_topic.split())

                # This also requires the actual LDA model object (lda_model_4_topics) to be available.
                # Since it's not saved and loaded, we can't accurately predict the topic. 
                # We will output a placeholder for topic prediction.
                predicted_topic = "(LDA Model and Dictionary need to be saved/loaded for accurate topic prediction in standalone app)"

            except Exception as e:
                predicted_topic = f"Error in topic prediction setup: {e}"
                st.warning("Warning: LDA model or dictionary could not be loaded/reconstructed for topic prediction. Please ensure they are saved and loaded correctly in a standalone app.")

            st.subheader("Analysis Results:")
            st.write(f"**Cleaned Review (for LSTM):** {cleaned_for_sentiment}")
            st.write(f"**Predicted Sentiment (LSTM):** {predicted_sentiment}")
            st.write(f"**Predicted Topic (LDA):** {predicted_topic}")

    else:
        st.warning("Please enter some text to analyze.")
