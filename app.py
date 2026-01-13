import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# SETUP HALAMAN
# =========================
st.set_page_config(page_title="NLP Analisis Sentimen Film", layout="wide")
st.title("🎬 NLP Analisis Sentimen Ulasan Film")
st.write("Pipeline: Case Folding → Cleaning → Tokenizing → Stopword Removal → Stemming (Sastrawi)")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_moview_review_1500.csv")
    return df

df = load_data()

# =========================
# VALIDASI KOLOM (ERROR HANDLING YANG BENAR)
# =========================
required_columns = ['Text Tweet', 'Sentiment']
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ File CSV harus memiliki kolom '{col}'")
        st.write(f"Kolom yang tersedia: {list(df.columns)}")
        st.stop()

# =========================
# STOPWORDS & STEMMER
# =========================
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =========================
# FUNGSI PREPROCESSING
# =========================
def preprocess(text):
    # 1. Case Folding
    text = text.lower()

    # 2. Cleaning
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenizing
    tokens = text.split()

    # 4. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Stemming (WAJIB)
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# =========================
# PREPROCESS DATA
# =========================
with st.spinner('Sedang memproses data...'):
    df['hasil_preprocessing'] = df['Text Tweet'].astype(str).apply(preprocess)
    st.success('✅ Preprocessing selesai!')

# =========================
# TAMPILKAN INFO DATASET
# =========================
st.write(f"📊 **Total data:** {len(df)} baris")
st.write(f"📋 **Kolom tersedia:** {list(df.columns)}")

# =========================
# TAMPILKAN DATA
# =========================
st.subheader("📄 Dataset Asli")
st.dataframe(df[['Text Tweet', 'Sentiment']].head(10))

st.subheader("🔄 Hasil Preprocessing (Stemming Sastrawi)")
st.dataframe(df[['hasil_preprocessing', 'Sentiment']].head(10))

# =========================
# MACHINE LEARNING PIPELINE
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# TF-IDF VECTORIZATION
# =========================
st.subheader("🧮 TF-IDF Vectorization")

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['hasil_preprocessing'])
y = df['Sentiment']

st.write(f"📐 Bentuk data TF-IDF: {X.shape}")

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# =========================
# EVALUATION
# =========================
st.subheader("📈 Evaluasi Model")
st.write(f"🎯 **Akurasi Model:** {accuracy:.2f}")

st.text("📊 Classification Report")
st.text(classification_report(y_test, y_pred))

# =========================
# PREDIKSI MANUAL (INPUT USER)
# =========================
st.subheader("🧪 Prediksi Sentimen Teks")

user_input = st.text_area(
    "Masukkan teks ulasan film:",
    placeholder="Contoh: Film ini sangat membosankan dan ceritanya tidak masuk akal"
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        processed_text = preprocess(user_input)
        vectorized_text = tfidf.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("😡 Sentimen NEGATIF")
        else:
            st.success("😊 Sentimen POSITIF")

        st.write("🧹 Hasil Preprocessing:")
        st.code(processed_text)

# =========================
# VISUALISASI SENTIMEN
# =========================
st.subheader("📊 Visualisasi Sentimen Dataset")

sentiment_counts = df['Sentiment'].value_counts()

fig, ax = plt.subplots()
ax.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
ax.axis('equal')

st.pyplot(fig)

# =========================
# WORD CLOUD
# =========================
st.subheader("☁️ WordCloud Ulasan Film")

all_text = " ".join(df['hasil_preprocessing'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(all_text)

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')

st.pyplot(fig_wc)
