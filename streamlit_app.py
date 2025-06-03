import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Stres Tahmin Uygulaması", layout="centered")
st.title("🧠 Stres Tahmin Uygulaması")
st.markdown("Bu uygulama, belirli psikolojik ve davranışsal ölçütlere göre kişinin stresli olup olmadığını tahmin eder.")

features = ['cesd', 'mbi_ex', 'mbi_ea', 'health', 'mbi_cy']
descriptions = [
    '70 üzeri → yüksek depresyon riski',
    '80 üzeri → tükenmişlik ihtimali',
    '60 üzeri → empati kaybı olabilir',
    '70 üzeri → sağlık algısı düşük olabilir',
    '80 üzeri → duyarsızlaşma riski'
]

st.sidebar.header("🔧 Girdi Değerleri")
user_values = []
for i in range(len(features)):
    val = st.sidebar.slider(f"{features[i]}", 0, 100, 50)
    st.sidebar.caption(descriptions[i])
    user_values.append(val)

input_data = np.array(user_values).reshape(1, -1)

# Model ve scaler yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Normalize ve tahmin
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Sonuç
st.subheader("📊 Tahmin Sonucu:")
if prediction == 1:
    st.error("🔴 Tahmin: **Stresli**")
else:
    st.success("🟢 Tahmin: **Stresli Değil**")

st.caption("Model: KNN (Korelasyonla seçilmiş 5 özellik)")

