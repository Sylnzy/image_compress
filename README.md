# Aplikasi Kompresi Gambar

Aplikasi web sederhana untuk mengompresi gambar dengan Flask dan MySQL.

## Fitur

- Upload dan kompresi gambar dengan kualitas yang dapat disesuaikan
- Penyimpanan data kompresi di database MySQL
- Tampilan riwayat kompresi
- Perhitungan rasio kompresi

## Struktur Project

```
app.py                  # File utama aplikasi Flask
requirements.txt        # Dependensi Python
static/                 # File statis (CSS, JS, dan gambar hasil kompresi)
  ├── css/              
  │   └── style.css     # File CSS
  ├── js/               # File JavaScript (jika ada)
  └── images/           # Folder untuk menyimpan hasil kompresi
templates/              # File template HTML
  ├── base.html         # Template dasar
  ├── index.html        # Halaman utama
  ├── result.html       # Halaman hasil kompresi
  └── history.html      # Halaman riwayat kompresi
```

## Setup Database MySQL

1. Buat database MySQL baru:
   ```sql
   CREATE DATABASE image_compress;
   ```

2. Pastikan konfigurasi database di `app.py` sesuai dengan pengaturan lokal Anda:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/image_compress'
   ```
   Ganti `username`, `password` dengan kredensial MySQL Anda.

## Instalasi

1. Clone repository ini

2. Buat dan aktifkan virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:
   ```
   python app.py
   ```

5. Buka browser dan akses `http://127.0.0.1:5000/`

## Mengubah Database

Jika Anda ingin menggunakan database MySQL dengan username dan password yang berbeda, edit baris berikut di `app.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/image_compress'
```

Ubah menjadi:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/image_compress'
```

## Kontributor

- [Nama Anda]