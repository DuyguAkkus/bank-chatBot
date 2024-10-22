import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore # CORS'u içe aktar

app = Flask(__name__)
CORS(app)  # CORS'u uygulamanıza ekleyin

# Veri yükleme
data = pd.read_csv('kitapım.txt', sep=';', header=None, names=['kategori', 'soru', 'yanit'], encoding='utf-8')

# Soru ve cevapları ayır
sorular = data['soru'].tolist()
cevaplar = data['yanit'].tolist()

# TF-IDF ile soruları vektörize et
vectorizer = TfidfVectorizer()
soru_vectors = vectorizer.fit_transform(sorular)

# Chatbot fonksiyonu
@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json['question']
    user_vector = vectorizer.transform([user_question])
    
    # Cosine Similarity hesaplama
    similarities = cosine_similarity(user_vector, soru_vectors)
    en_yakin_soru_index = similarities.argmax()

        # Eğer benzerlik çok düşükse veya yanıtı bulamadıysak, özel bir yanıt döndür
    if similarities[0][en_yakin_soru_index] < 0.1:  # Benzerlik eşiğini kendinize göre ayarlayabilirsiniz
        return jsonify({'answer': "Üzgünüm, bu soruya yanıt veremiyorum. Başka bir şey sormak ister misiniz?"})
    
    # En yakın cevabı bul
    answer = cevaplar[en_yakin_soru_index]
   

    # Logları yazdır
    print(f"User question: {user_question}")
    print(f"Similarities: {similarities}")
    print(f"En yakın soru indeksi: {en_yakin_soru_index}")
    
    return jsonify({'answer': answer})

# Eğitim verileri ile test etme
def evaluate_model():
    doğru_tahminler = 0

    for index, soru in enumerate(sorular):
        user_vector = vectorizer.transform([soru])
        similarities = cosine_similarity(user_vector, soru_vectors)
        en_yakin_soru_index = similarities.argmax()
        predicted_answer = cevaplar[en_yakin_soru_index]
        
        if predicted_answer == cevaplar[index]:
            doğru_tahminler += 1

    başarı_oranı = doğru_tahminler / len(sorular) * 100
    print(f"Başarı Oranı: {başarı_oranı}%")

if __name__ == '__main__':
    evaluate_model()  # Modeli değerlendir
    app.run(debug=True)

#   KODU ÇALIŞTIR 
#  cd backend 
#  source yenmv/bin/activate 
#  python chatbot_model.py