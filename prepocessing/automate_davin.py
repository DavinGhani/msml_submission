# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# --- 1. Mendefinisikan Path ---

# Path ke file mentah (raw)
# os.path.join('..', 'dataset_raw', 'heart.csv')
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset_raw', 'heart.csv')

# Path untuk menyimpan data yang sudah diproses 
# Kita akan buat folder baru untuk ini
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset_preprocessing')

# Buat folder 'dataset_preprocessing' jika belum ada
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print(f"Path data mentah: {RAW_DATA_PATH}")
print(f"Direktori data proses: {PROCESSED_DATA_DIR}")

# --- 2. Fungsi Preprocessing Utama ---

def preprocess(raw_data_path, processed_data_dir):
    """
    Fungsi ini melakukan preprocessing lengkap:
    1. Memuat data
    2. Membersihkan duplikat
    3. Memisahkan fitur (X) dan target (y)
    4. Membagi data (train/test split)
    5. Menerapkan scaling (numerik) dan encoding (kategorikal)
    6. Menyimpan hasil data yang sudah diproses
    """
    
    print("Memulai preprocessing...")

    # --- 4.1 & 4.2: Load & Cleaning ---
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Data mentah dimuat: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {raw_data_path}")
        return

    # Hapus duplikat
    df.drop_duplicates(inplace=True)
    print(f"Data setelah dihapus duplikat: {df.shape}")

    # --- 4.3: Pemisahan Fitur (X) dan Target (y) ---
    X = df.drop('target', axis=1)
    y = df['target']

    # --- 4.4: Pembagian Data (Train-Test Split) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Data dibagi: {X_train.shape} train, {X_test.shape} test")

    # --- 4.5: Definisi Kolom & Pipeline Preprocessing ---
    # Definisikan nama kolom 
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Pipeline untuk data numerik
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk data kategorikal
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan pipeline (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 4.6: Menerapkan Pipeline ke Data ---
    print("Menerapkan preprocessing (fit_transform) ke X_train...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    print("Menerapkan preprocessing (transform) ke X_test...")
    X_test_preprocessed = preprocessor.transform(X_test)

    # --- 7. Menyimpan Data yang Sudah Diproses ---
    # Kita perlu mendapatkan nama kolom setelah OneHotEncoding
    # Ini penting agar file CSV kita punya header
    
    # Ambil nama fitur kategorikal baru dari OneHotEncoder
    ohe_feature_names = preprocessor.named_transformers_['cat'] \
                                    .named_steps['onehot'] \
                                    .get_feature_names_out(categorical_features)
    
    # Gabungkan nama fitur numerik + nama fitur OHE baru
    all_feature_names = numerical_features + list(ohe_feature_names)

    # Konversi hasil (sparse matrix) ke DataFrame Pandas
    X_train_processed_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)
    X_test_processed_df = pd.DataFrame(X_test_preprocessed, columns=all_feature_names)

    # Gabungkan kembali X dan y untuk disimpan (praktik umum)
    train_data = pd.concat([X_train_processed_df, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_processed_df, y_test.reset_index(drop=True)], axis=1)

    # Simpan ke file CSV di folder 'namadataset_preprocessing'
    train_data_path = os.path.join(processed_data_dir, 'train_processed.csv')
    test_data_path = os.path.join(processed_data_dir, 'test_processed.csv')
    
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    print(f"Data train diproses disimpan di: {train_data_path}")
    print(f"Data test diproses disimpan di: {test_data_path}")
    print("Preprocessing selesai.")


# --- 3. Menjalankan Fungsi ---
if __name__ == '__main__':
    # Ini memastikan kode di bawah hanya berjalan saat file ini dieksekusi
    # langsung, bukan saat di-import.
    preprocess(raw_data_path=RAW_DATA_PATH, processed_data_dir=PROCESSED_DATA_DIR)