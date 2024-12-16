import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import requests
from collections import Counter
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
    
# Muat data kamus
df_kamus_komen1 = pd.read_excel('data_komen_mundjidah_clean.xlsx')  # Kamus 1
df_kamus_komen3 = pd.read_excel('data_komen_warsubi_clean-v1.xlsx')  # Kamus 3

# Daftar kata kunci negatif dan positif
negative_keywords_model1 = ["pilih nomor dua", "nomor dua", "buruk", "jelek", "✌️", "dua", "jalan rusak", "leren", "perubahan", "ganti bupati", "warsa", "abah", "janji manis", "omong tok", "nyocot", "bacot"]
negative_keywords_model2 = ["pilih nomor satu", "nomor satu", "buruk", "jelek","☝️"]
negative_keywords_model3 = ["buruk", "jelek", "☝️", "golput", "serang", "mundjidah", "janji manis", "omong tok", "nyocot", "bacot", "carmuk","cari muka"]

positive_keywords_model1 = ["semoga menang", "semoga", "baik", "bagus", "terbaik", "semangat", "mundjidah", "amin", "gas"]
positive_keywords_model2 = ["hebat", "luar biasa", "bagus", "terbaik", "memilih dengan tepat", "all in abah subi", "pilih warsubi"]
positive_keywords_model3 = ["hebat", "luar biasa", "bagus", "terbaik", "memilih dengan tepat", "all in abah subi", "pilih warsubi", "coblos", "dukung", "pilih", "semangat" , "allahuakbar","subhanallah","gus kautsar", "pemimpin", "gus", "pendherek",
                            "salam dua jari", "pemimpin baru", "alhamdulillah","salam","sowan", "waalaikumsalam", "tambah maju", "tambah sejahtera", "makin maju", "makin sejahtera", "makin apik","hadir", "sip", "jos", "mantap bah",
                            "warsa", "warsubi", "warsa bupatiku", "setuju", "dukung abah", "abah", "dua", "nomor dua", "amin", "gas", "ayo dukung", "warsubi tok", "semoga menang", "warsa ae", "warsa ae liane up", "tiang sae","bantu","beri","kasih",
                            "selamat","pasti menang", "assalamualaikum",  "unggul", "telak", "perubahan", "semoga", "warga sejahtera", "semakin sejahtera", "tambah apik", "ganti bupati","ngayomi", "alhamdulillah","barokalloh", "pilih abah", "pilih warsa",
                            "aamiin", "bismilah", "pasti menang", "bismillah", "aamiin", "calon pemimpin", "dukung abah subi", "alhamdulillah", "masyaallah","mashaallah", "menang", "pemimpin", "warsah", "lanjutkan abah", "lanjutkan"
                            "semangat", "optimis", "semoga", "yakin", "amanah", "mantap", "mantab", "komitmen", "mengayomi","merangkul","bupati","calon bupati","bupati", "bukan pencitraan", "dermawan", "bantuan", "no dua", "no ✌️"]

# Fungsi untuk memuat kamus normalisasi dari file lokal
def load_normalization_dict(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            normalization_dict = {}
            for line in lines:
                line = line.strip()
                if ':' in line:  # Memastikan format key:value
                    key, value = line.split(':', 1)  # Pisahkan berdasarkan ':'
                    key = key.strip('"')  # Hapus tanda kutip pada key
                    value = value.strip('",')  # Hapus tanda kutip dan koma pada value
                    normalization_dict[key.strip()] = value.strip()
            return normalization_dict
    except Exception as e:
        st.error(f"Gagal memuat kamus normalisasi: {e}")
        return {}

# Muat kamus normalisasi dari file lokal
normalization_file = "slang.txt"
normalization_dict = load_normalization_dict(normalization_file)

# Fungsi untuk melakukan normalisasi teks
def normalize_text(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return " ".join(normalized_words)
    
# Fungsi untuk membersihkan komentar dari username
def get_known_usernames(data):
    # Cek apakah kolom "Author" atau "Username" ada
    if "Author" in data.columns:
        return set(data["Author"].str.strip().str.lower())
    elif "Username" in data.columns:
        return set(data["Username"].str.strip().str.lower())
    elif "Nama Akun" in data.columns:
        return set(data["Nama Akun"].str.strip().str.lower())
    else:
        # Jika tidak ada kolom, kembalikan set kosong
        return set()

def remove_usernames(comment, usernames):
    for username in usernames:
        pattern = rf'\b{re.escape(username)}\b'
        comment = re.sub(pattern, '', comment, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', comment.strip())
    
# Fungsi untuk membersihkan teks
def clean_text(text):
    text = str(text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\b(01|1)\b', 'satu', text)
    text = re.sub(r'\b(02|2)\b', 'dua', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = text.lower().strip()
    return re.sub(r'[^a-zA-Z0-9✌️☝️ ]', '', text)


# Fungsi untuk memperbarui kamus
def update_kamus(file_path, new_data):
    try:
        existing_data = pd.read_excel(file_path)  # Muat data kamus yang ada
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=["Comment"])  # Hindari duplikasi
        combined_data.to_excel(file_path, index=False)  # Simpan kembali ke file
        st.success(f"Kamus berhasil diperbarui dengan {len(new_data)} data baru.")
    except Exception as e:
        st.error(f"Gagal memperbarui kamus: {e}")


# Tambahkan opsi di sidebar
menu = st.sidebar.selectbox("Pilih Menu", ["Klasifikasi Sentimen", "Editor Kamus"])

if menu == "Klasifikasi Sentimen":
    # Streamlit app
    st.title("Aplikasi Klasifikasi Sentimen dan Brand Attitude")
    
    # Pilihan model
    model_choice = st.selectbox("Pilih Model:", ["Model Mundjidah", "Model Warsubi V1"])
    
    # Upload file
    uploaded_file = st.file_uploader("Upload file Excel atau CSV", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            # Baca file yang diunggah
            if uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
    
            # Bersihkan data
            data.dropna(how='all', inplace=True)
            data['Comment'] = data['Comment'].fillna('')
            data = data[data['Comment'].str.strip() != '']
            
            # Proses pembersihan teks termasuk normalisasi
            known_usernames = get_known_usernames(data)
            data["Cleaned_Text"] = data["Comment"].apply(lambda x: remove_usernames(x, known_usernames))
            data["Cleaned_Text"] = data["Cleaned_Text"].apply(lambda x: normalize_text(clean_text(x), normalization_dict))

            # Konfigurasi model berdasarkan pilihan
            if model_choice == "Model Mundjidah":
                sentiment_model_path = "mundjidah-model.h5"
                ba_model_path = "ba-mundjidah-model.h5"
                positive_keywords = ["semoga menang", "semoga", "baik", "bagus", "terbaik", "semangat", "mundjidah", "amin", "gas"]
                negative_keywords = ["pilih nomor dua", "nomor dua", "buruk", "jelek", "✌️", "dua", "jalan rusak", "dalan rusak", "leren", "perubahan", "ganti bupati"]

            elif model_choice == "Model Warsubi V1":
                sentiment_model_path = "warsa-model.h5"
                ba_model_path = "ba-warsa-model.h5"
                positive_keywords = ["hebat", "luar biasa", "bagus", "terbaik", "memilih dengan tepat", "all in abah subi", "pilih warsubi", "dua", "✌️", "abah", "sae"]
                negative_keywords = ["pilih nomor satu", "nomor satu", "buruk", "jelek", "☝️", "golput ae", "serang", "mundjidah", "janji manis", "nyocot", "bacot", "carmuk"]

            else:  # Tambahan untuk model lain
                sentiment_model_path = "warsubi-v2-model.h5"
                ba_model_path = "ba-warsubi-v2-model.h5"
                positive_keywords = ["hebat", "luar biasa", "bagus", "terbaik", "coblos", "dukung", "pilih", "semangat"]
                negative_keywords = ["golput ae", "serang", "mundjidah", "janji manis", "nyocot", "bacot", "carmuk"]

            PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'
            # Load model sentimen
            try:
                sentiment_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=3)
                sentiment_model.load_weights(sentiment_model_path)
                tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
            except Exception as e:
                st.error(f"Gagal memuat model sentimen: {e}")
                st.stop()
    
            # Fungsi prediksi sentimen dengan tambahan pencocokan keyword
            def predict_with_sentiment_model(text):
                # Pencocokan keyword
                if any(keyword.lower() in text.lower() for keyword in positive_keywords):
                    return 'positive'
                elif any(keyword.lower() in text.lower() for keyword in negative_keywords):
                    return 'negative'

                # Prediksi menggunakan model jika tidak ada keyword yang cocok
                inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
                outputs = sentiment_model(inputs)
                logits = outputs.logits
                predicted_label = tf.argmax(logits, axis=1).numpy()[0]
                return ['negative', 'positive', 'neutral'][predicted_label]

            data['Sentimen_Prediksi'] = data['Cleaned_Text'].apply(predict_with_sentiment_model)
    
            # Load model Brand Attitude
            try:
                ba_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=4)
                ba_model.load_weights(ba_model_path)
            except Exception as e:
                st.error(f"Gagal memuat model Brand Attitude: {e}")
                st.stop()
    
            # Daftar keyword untuk masing-masing kategori
            keywords = {
                "Co-Optimism": ["semoga sehat selalu", "semoga sukses", "lanjutkan", "semangat", "sehat", "setuju", "ayo", "selamat", "sukses", 
                                "semoga", "berharap", "mugo", "lebih maju", "optimis jombang satu",  "bangga", "saget", "doa", "tambah maju", 
                                "lebih maju", "tambah makmur", "tambah sejahtera", "majukan", "harap", "berharap", "menginginkan", "ingin", 
                                "mendoakan", "sae bah", "bismilah", "cocok", "umkm maju", "butuh perubahan", "butuh ganti bupati", "memakmurkan", 
                                "makmur", "buka lapangan kerja", "lancar", "lancar terus", "mugi", "bantuan", "sembako", "lebih baik", "tambah apik", 
                                "sae", "tambah sae", "jombang maju bersama warsa", "jombang maju", "sejahtera", "yakin", "makin", 
                                "optimis", "salam","jombang sejahtera","tambah sejahtera", "butuh pemimpin","bismillah", "warsa menang",
                                "menanti pemimpin", "bakalan maju", "bakalan sejahtera", "bakalan sukses","yakin", "majukan", "majulah", "doakan"],
                
                "Co-Support": ["siap dukung", "all in", "menyala", "siap", "dukung", "gas", "warsa", "menang", "coblos", "coblos dua", 
                            "ayo", "pilih dua", "pilih", "wonge abah", "warsubi tok", "merangkul", "program", "konkrit", "wong apik", 
                            "baik", "niat apik", "merakyat", "mengayomi", "komitmen", "merangkul", "mendengar", "dengar", "panggah abah", 
                            "panggah warsa", "antusias", "komitmen", "kebersamaan", "dukung abah", "dengan abah", "program konkrit", "abah satu", 
                            "jombang satu", "orang baik", "pilih abah", "pilih warsa", "wonge abah", "ngopeni ngayomi mumpuni", "melu", 
                            "tambah adem", "tambah sejuk", "dukung usaha", "no dua", "dukung umkm", "dukung ekonomi", "pendherek", "penderek", 
                            "pengikut", "bismilah abah", "abah dua", "hadir support", "nggih", "turun tangan", "membantu", "bertindak", 
                            "melaju", "program", "membantu", "bupati", "joss", "top", "jombang maju", "wayae", "wayahe", "maju", "mantap", 
                            "abah", "bah", "ganti bupati", "sodaqoh", "wayahe ganti", "ganti", "meledak", "menyala", "dibutuhkan", "kawal", 
                            "membara", "seru", "keren", "mantap", "istimewa", "ayo", "layak", "al in", "makin raket", "kerja nyata", 
                            "selalu dihati", "pangah abah", "pangah warsa", "kebersaman", "dermawan", "sat set", "wat wet", "panggah abah", 
                            "panggah warsa", "pangah warsa", "pangah", "wonge abah", "positif menang", "pemimpin", "wong mu"]
                
            }

            # Fungsi prediksi BA dengan tambahan pencocokan keyword
            def predict_ba_with_model(text):
                # Mengecek apakah teks mengandung kata-kata kunci dari kategori Co-Support atau Co-Optimism
                for label, keywords_list in keywords.items():
                    if any(keyword.lower() in text.lower() for keyword in keywords_list):
                        return label  # Jika ada keyword yang cocok, kembalikan label yang sesuai
                
                # Jika tidak ada keyword yang cocok, gunakan model untuk prediksi
                inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
                outputs = ba_model(inputs)
                logits = outputs.logits
                predicted_label = tf.argmax(logits, axis=1).numpy()[0]
                return ['Co-Likes', 'Co-Support', 'Co-Optimism', 'Co-Negative'][predicted_label]

            data['Brand_Attitude'] = data['Cleaned_Text'].apply(predict_ba_with_model)
    
            # Tambahkan "Co-Negative" jika Sentimen_Prediksi adalah "negative"
            data['Brand_Attitude'] = data.apply(
                lambda row: "Co-Negative" if row['Sentimen_Prediksi'] == 'negative' else row['Brand_Attitude'], axis=1
            )

            # Jika Sentimen_Prediksi bukan "negative", tapi Brand_Attitude berisi "Co-Negative", ubah jadi "Co-Likes"
            data['Brand_Attitude'] = data.apply(
                lambda row: "Co-Likes" if row['Sentimen_Prediksi'] != 'negative' and row['Brand_Attitude'] == 'Co-Negative' else row['Brand_Attitude'], axis=1
            )
    
            # Tampilkan hasil
            st.write("Hasil Klasifikasi Sentimen:")
            st.dataframe(data[['Comment', 'Cleaned_Text', 'Sentimen_Prediksi', 'Brand_Attitude']])

            # Distribusi sentimen
            sentiment_counts = data['Sentimen_Prediksi'].value_counts()            
            # Hitung jumlah sentimen
            total_positive = sentiment_counts.get('positive', 0)
            total_negative = sentiment_counts.get('negative', 0)
            total_neutral = sentiment_counts.get('neutral', 0)
            
            # Tampilkan total jumlah sentimen
            st.write(f"**Total Sentimen Positif:** {total_positive}")
            st.write(f"**Total Sentimen Negatif:** {total_negative}")
            st.write(f"**Total Sentimen Netral:** {total_neutral}")
    
            # Distribusi level komentar
            st.write("Distribusi Level Komentar:")
            level_counts = data['Brand_Attitude'].value_counts()
            total_co_likes = level_counts.get('Co-Likes', 0)
            total_co_support = level_counts.get('Co-Support', 0)
            total_co_optimism = level_counts.get('Co-Optimism', 0)
            total_co_negative = level_counts.get('Co-Negative', 0)
            
            # Tampilkan total jumlah sentimen
            st.write(f"**Total BA Co-Likes:** {total_co_likes}")
            st.write(f"**Total BA Co-Support:** {total_co_support}")
            st.write(f"**Total BA Co-Optimism:** {total_co_optimism}")
            st.write(f"**Total BA Co-Negative:** {total_co_negative}")
            
            # Visualisasi distribusi sentimen
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Distribusi Sentimen")
            ax.set_xlabel("Sentimen")
            ax.set_ylabel("Jumlah Komentar")
            st.pyplot(fig)
            
            # Tampilkan jumlah setiap kategori
            st.bar_chart(level_counts)
    
            def generate_wordcloud(text):
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=200,
                    colormap='viridis'
                ).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                return fig
    
            st.write("WordCloud Berdasarkan Sentimen:")
            for sentiment in ['positive', 'negative', 'neutral']:
                text = " ".join(data[data['Sentimen_Prediksi'] == sentiment]['Cleaned_Text'].tolist())
                if text:
                    st.write(f"WordCloud untuk Sentimen {sentiment.capitalize()}:")
                    st.pyplot(generate_wordcloud(text))
    
            # Tampilkan kalimat berdasarkan sentimen
            st.write("Kalimat Berdasarkan Sentimen:")
            st.write("### Kalimat Positif")
            st.write(data[data['Sentimen_Prediksi'] == 'positive']['Comment'].tolist())
            
            st.write("### Kalimat Negatif")
            st.write(data[data['Sentimen_Prediksi'] == 'negative']['Comment'].tolist())
            
            st.write("### Kalimat Netral")
            st.write(data[data['Sentimen_Prediksi'] == 'neutral']['Comment'].tolist())
            
            # Fungsi untuk tokenisasi teks
            def tokenize_text(text):
                """Membersihkan dan memisahkan teks menjadi kata-kata."""
                # Hilangkan tanda baca, konversi ke huruf kecil, dan split
                words = text.lower().replace('.', '').replace(',', '').split()
                return words
            
            # Fungsi untuk menghitung frekuensi kata
            def get_word_frequencies(data, column):
                """Menghitung frekuensi kata berdasarkan kolom teks tertentu."""
                all_words = []
                for text in data[column]:
                    all_words.extend(tokenize_text(text))
                return Counter(all_words)
            
            # Filter data berdasarkan kategori
            neutral_data = data[data['Sentimen_Prediksi'] == 'neutral']
            co_likes_data = data[data['Brand_Attitude'] == 'Co-Likes']
            
            # Hitung frekuensi kata untuk masing-masing kategori
            neutral_word_counts = get_word_frequencies(neutral_data, 'Cleaned_Text')
            co_likes_word_counts = get_word_frequencies(co_likes_data, 'Cleaned_Text')
            
            # Visualisasi chart untuk kata-kata di sentimen neutral
            st.write("### Top Kata di Sentimen Neutral")
            neutral_most_common = neutral_word_counts.most_common(10)
            neutral_words, neutral_counts = zip(*neutral_most_common)
            
            plt.figure(figsize=(10, 6))
            plt.barh(neutral_words, neutral_counts, color='skyblue')
            plt.xlabel('Frequency')
            plt.ylabel('Words')
            plt.title('Top Words in Neutral Sentiment')
            plt.gca().invert_yaxis()
            st.pyplot(plt)
            
            # Visualisasi chart untuk kata-kata di Co-Likes
            st.write("### Top Kata di BA Co-Likes")
            co_likes_most_common = co_likes_word_counts.most_common(10)
            co_likes_words, co_likes_counts = zip(*co_likes_most_common)
            
            plt.figure(figsize=(10, 6))
            plt.barh(co_likes_words, co_likes_counts, color='lightgreen')
            plt.xlabel('Frequency')
            plt.ylabel('Words')
            plt.title('Top Words in Co-Likes Category')
            plt.gca().invert_yaxis()
            st.pyplot(plt)
            
            # Siapkan data untuk diperbarui
            new_data = data[['Comment', 'Cleaned_Text', 'Sentimen_Prediksi']].copy()
            new_data.rename(columns={'Sentimen_Prediksi': 'Sentimen_Aktual'}, inplace=True)
            
            # Fungsi untuk mencari komentar yang mirip
            def find_similar_comments(data, query_text, top_n=5):
                # Membuat representasi TF-IDF dari teks
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(data['Cleaned_Text'])
                
                # Mencari query dalam database
                query_tfidf = vectorizer.transform([query_text])
                
                # Menghitung cosine similarity
                similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
                
                # Menambahkan similarity ke dataframe
                data['similarity'] = similarity_scores[0]
                
                # Mengurutkan berdasarkan similarity tertinggi
                similar_comments = data.sort_values(by='similarity', ascending=False).head(top_n)
                
                return similar_comments

            # Menampilkan data komentar yang mirip
            st.write("Komentar yang Mirip dengan Sentimen yang Akan Diperbarui")
            similar_comments = find_similar_comments(data, "Komentar yang ingin diubah sentimennya", top_n=5)
            st.dataframe(similar_comments[['Comment', 'Cleaned_Text', 'Sentimen_Prediksi', 'similarity']])

            # Menampilkan kolom input untuk mengubah sentimen dan brand attitude
            new_sentiment = st.selectbox("Pilih Sentimen Baru", ['positive', 'negative', 'neutral'])
            new_brand_attitude = st.selectbox("Pilih Brand Attitude Baru", ['Co-Likes', 'Co-Support', 'Co-Optimism', 'Co-Negative'])

            # Tombol untuk memperbarui sentimen dan brand attitude
            if st.button("Perbarui Sentimen dan Brand Attitude"):
                updated_comments = similar_comments.copy()
                updated_comments['Sentimen_Aktual'] = new_sentiment
                updated_comments['Brand_Attitude'] = new_brand_attitude
                
                # Update data di database atau dataframe
                # Misalnya, jika data disimpan dalam DataFrame `data`
                for index, row in updated_comments.iterrows():
                    data.loc[data['Cleaned_Text'] == row['Cleaned_Text'], 'Sentimen_Aktual'] = row['Sentimen_Aktual']
                    data.loc[data['Cleaned_Text'] == row['Cleaned_Text'], 'Brand_Attitude'] = row['Brand_Attitude']
                
                st.success("Sentimen dan Brand Attitude berhasil diperbarui!")
                
                # # Menyimpan setiap baris ke dalam database
                # for index, row in new_data.iterrows():
                #     comment = row['Comment']
                #     cleaned_text = row['Cleaned_Text']
                #     sentimen_aktual = row['Sentimen_Aktual']
                    
                # # Tambahkan tombol untuk memperbarui kamus
                # if st.button("Perbarui Kamus"):
                #     new_data = data[['Comment', 'Cleaned_Text', 'Sentimen_Prediksi']].copy()
                #     new_data.rename(columns={'Sentimen_Prediksi': 'Sentimen_Aktual'}, inplace=True)
                #     update_kamus(selected_file, new_data)
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# Definisikan hyperparameter
PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

# Fungsi untuk melatih ulang model
def retrain_model(kamus_data):
    # Siapkan data
    X = kamus_data['Cleaned_Text']
    y = kamus_data['Sentimen_Aktual']

    # Mengganti label secara manual (tanpa LabelEncoder)
    y = y.apply(lambda label: 0 if label == 'negative' else (1 if label == 'positive' else 2))

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenisasi dan padding (BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, max_length=128, return_tensors='tf')
    X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, max_length=128, return_tensors='tf')

    model_path = ''
    if kamus_data == "data_komen_mundjidah_clean.xlsx":
        model_path = 'update_mundjidah-model.h5'
    elif kamus_data == "data_komen_warsubi_clean-v1.xlsx":
        model_path = 'update_warsubi-model.h5'
        
    # Load model BERT untuk Sequence Classification
    bert_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=3)

    # Tentukan optimizer dan loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # Compile model
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Latih model
    bert_model.fit(
        X_train_tokens['input_ids'], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_tokens['input_ids'], y_test)
    )

    # Simpan model yang sudah dilatih
    bert_model.save(model_path)

    st.success("Model berhasil dilatih ulang dan disimpan!")

if menu == "Editor Kamus":
    st.title("Editor Kamus")
    kamus_option = st.selectbox(
        "Pilih Kamus yang Ingin Diedit:",
        ["data_komen_mundjidah_clean.xlsx", "data_komen_warsubi_clean-v1.xlsx"]
    )

    # Validasi pilihan kamus
    if kamus_option in ["data_komen_mundjidah_clean.xlsx", "data_komen_warsubi_clean-v1.xlsx"]:
        # Muat file kamus dari Excel
        try:
            kamus_data = pd.read_excel(kamus_option)

            st.write("Kamus Saat Ini:")
            # Tampilkan tabel yang dapat diedit
            edited_data = st.data_editor(
                kamus_data,
                use_container_width=True,
                height=500
            )

            # Tombol untuk menyimpan perubahan
            if st.button("Simpan Perubahan"):
                edited_data.to_excel(kamus_option, index=False)
                st.success("Perubahan berhasil disimpan ke file Excel!")

            # Tombol untuk retrain model
            if st.button("Retrain Model"):
                retrain_model(kamus_data)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat atau menyimpan kamus: {e}")
