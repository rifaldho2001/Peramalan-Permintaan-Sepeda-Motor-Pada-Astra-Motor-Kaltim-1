import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# Membaca save model
best_models = {
    'ADV': 'Model SAV/ADV.sav',
    'BEAT': 'Model SAV/BEAT.sav',
    'BEAT STREET': 'Model SAV/BEAT STREET.sav',
    'GENIO': 'Model SAV/GENIO.sav',
    'PCX': 'Model SAV/PCX.sav',
    'SCOOPY': 'Model SAV/SCOOPY.sav',
    'VARIO 125': 'Model SAV/VARIO 125.sav',
    'VARIO 160': 'Model SAV/VARIO 160.sav'
}

# Menggunakan save model


def load_model(model_name):
    model_filename = best_models[model_name]
    try:
        return joblib.load(model_filename)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Aplikasi Streamlit


# Define CSS styles
st.markdown(
    """
    <style>
    .sidebar {
        background-color: #c0d6df;
    }
    .dataframe {
        background-color: #dbe9ee;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    st.title(
        'PERAMALAN PERMINTAAN SEPEDA MOTOR PADA ASTRA MOTOR KALTIM 1 MENGGUNAKAN METODE SVR')

    # Membuat sidebar untuk input
    with st.sidebar:
        st.image("logoastra.png",
                 use_column_width=True)
        model_name = st.selectbox('Pilih Model :', list(best_models.keys()))
        time = st.selectbox('Pilih Waktu Peramalan :',
                            ('3 Bulan', '6 Bulan', '9 Bulan', '12 Bulan'))
        uploaded_file = st.file_uploader('Unggah file CSV', type=['csv'])
        generate_button = st.button('Generate')

    if generate_button:  # Tombol "Generate" ditekan
        if uploaded_file is not None:
            # Membaca file CSV yang telah diupload
            df = pd.read_csv(uploaded_file)
            # Memilih rentang waktu peramalan
            if time == '3 Bulan':
                df = df.iloc[:3]  # Keep the first 3 rows
            elif time == '6 Bulan':
                df = df.iloc[:6]  # Keep the first 6 rows
            elif time == '9 Bulan':
                df = df.iloc[:9]  # Keep the first 9 rows
            else:
                df = df.iloc[:12]  # Keep the first 12 rows
            # Menggunakan save model
            best_model = load_model(model_name)
            # Data preprocessing
            X = df.drop('Total', axis=1)
            y = df['Total']
            scaler_X = MinMaxScaler()
            X_normalized = scaler_X.fit_transform(X)
            scaler_y = MinMaxScaler()
            y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))
            # Prediksi
            y_pred = best_model.predict(X_normalized)
            # Transformasi data
            y_denorm = scaler_y.inverse_transform(y_normalized)
            y_pred_denorm = scaler_y.inverse_transform(
                y_pred.reshape(-1, 1))
            # Membuat dataframe untuk visualisasi
            prediction = pd.DataFrame(
                {'Actual': y_denorm.flatten(), 'Prediction': y_pred_denorm.flatten()})
            # Membuat rentang bulan
            months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                      "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
            # Visualisasi
            st.subheader('Grafik Peramalan:')
            fig, ax = plt.subplots(figsize=(13, 10))
            sns.lineplot(x=months[:len(df)],
                         y=prediction['Prediction'], ax=ax)
            ax.set_title('Peramalan')
            ax.set_xlabel('Tahun 2022')
            ax.set_ylabel('Jumlah Permohonan')
            st.pyplot(fig)
            # Menampilkan hasil peramalan
            st.subheader('Data Peramalan:')
            for i, (actual, pred) in enumerate(zip(y_denorm, y_pred_denorm)):
                month_name = months[i] if i < len(
                    months) else f"Bulan {i + 1}"
                st.write(f"{month_name} 2022 : {pred[0]:.6f}")


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
