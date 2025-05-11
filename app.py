from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify, session
import os
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime
import time
import threading
import uuid
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime, Float
from compression_methods import compress_image_dct, dwt_compress, compress_image_btc, compress_image_ambtc, compress_image_svd

app = Flask(__name__)
app.secret_key = 'kompresigambar123'  # Secret key untuk flash messages
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Konfigurasi database MySQL dengan PyMySQL driver
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/image_compress'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Menyimpan progress kompresi untuk setiap sesi
compression_progress = {}

# Model database untuk gambar yang dikompresi
class CompressedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    compressed_filename = db.Column(db.String(255), nullable=False)
    quality = db.Column(db.Integer, nullable=False)
    compress_date = db.Column(db.DateTime, default=datetime.now)
    file_size_before = db.Column(db.Float)
    file_size_after = db.Column(db.Float)
    compression_ratio = db.Column(db.Float)
    compression_method = db.Column(db.String(50), default='PIL')
    parameter_value = db.Column(db.String(50), nullable=True)

# Konfigurasi folder untuk menyimpan gambar hasil kompresi
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Daftar ekstensi file yang didukung untuk setiap metode kompresi
SUPPORTED_EXTENSIONS = {
    'PIL': ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'],
    'DCT': ['jpg', 'jpeg', 'png', 'bmp'],
    'DWT': ['jpg', 'jpeg', 'png', 'bmp'],
    'BTC': ['jpg', 'jpeg', 'png', 'bmp', 'pgm'],
    'AMBTC': ['jpg', 'jpeg', 'png', 'bmp', 'pgm'],
    'SVD': ['jpg', 'jpeg', 'png', 'bmp', 'pgm']
}

def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def is_file_compatible(filename, method):
    extension = get_file_extension(filename)
    return extension in SUPPORTED_EXTENSIONS[method]

@app.route('/')
def hello():
    # Ambil 5 kompresi terbaru untuk ditampilkan di halaman utama
    recent_compressions = CompressedImage.query.order_by(CompressedImage.compress_date.desc()).limit(5).all()
    
    compression_methods = [
        {
            'id': 'PIL', 
            'name': 'PIL Standard', 
            'description': 'Metode default dari library PIL',
            'suitable_for': 'Semua jenis gambar (JPG, PNG, JPEG, dll)',
            'best_for': 'Gambar berwarna dan grayscale',
            'file_limits': 'Semua ukuran',
            'icon': '📷',
            'color': '#4CAF50'
        },
        {
            'id': 'DCT', 
            'name': 'DCT (Discrete Cosine Transform)', 
            'description': 'Untuk kompresi kualitas tinggi, mirip JPEG',
            'suitable_for': 'JPG, JPEG, PNG',
            'best_for': 'Foto fotografi, detail tinggi',
            'file_limits': 'Ideal < 5MB',
            'icon': '🖼️',
            'color': '#2196F3'
        },
        {
            'id': 'DWT', 
            'name': 'DWT (Discrete Wavelet Transform)', 
            'description': 'Ukuran file kecil, efisiensi tinggi',
            'suitable_for': 'JPG, PNG, Grayscale',
            'best_for': 'Gambar dengan detail tepi penting',
            'file_limits': 'Terbaik < 3MB',
            'icon': '📊',
            'color': '#9C27B0'
        },
        {
            'id': 'BTC', 
            'name': 'BTC (Block Truncation Coding)', 
            'description': 'Kompresi cepat - ukuran kecil',
            'suitable_for': 'Grayscale, PGM',
            'best_for': 'Gambar grayscale dengan kontras tinggi',
            'file_limits': 'Terbaik < 1MB',
            'icon': '🔳',
            'color': '#FF9800'
        },
        {
            'id': 'AMBTC', 
            'name': 'AMBTC (Absolute Moment BTC)', 
            'description': 'Kompresi cepat & ringan',
            'suitable_for': 'HANYA GRAYSCALE, PGM, RAW',
            'best_for': 'Gambar grayscale, dokumen & text',
            'file_limits': 'Terbaik < 1MB',
            'icon': '📝',
            'color': '#795548'
        },
        {
            'id': 'SVD', 
            'name': 'SVD (Singular Value Decomposition)', 
            'description': 'Kompresi seimbang',
            'suitable_for': 'GRAYSCALE, PGM, RAW',
            'best_for': 'Gambar grayscale dengan area serupa',
            'file_limits': 'Terbaik < 2MB',
            'icon': '📐',
            'color': '#607D8B'
        }
    ]
    
    return render_template('index.html', 
                           title='Aplikasi Kompresi Gambar', 
                           recent_compressions=recent_compressions,
                           compression_methods=compression_methods)

@app.route('/compress', methods=['POST'])
def compress_image():
    if 'image' not in request.files:
        flash('Tidak ada file gambar yang dipilih', 'error')
        return redirect(url_for('hello'))
    
    file = request.files['image']
    if file.filename == '':
        flash('Tidak ada file gambar yang dipilih', 'error')
        return redirect(url_for('hello'))
    
    method = request.form.get('method', 'PIL')
    quality = int(request.form.get('quality', 80))
    parameter = request.form.get('parameter', None)
    
    # Validasi kompatibilitas file
    if not is_file_compatible(file.filename, method):
        flash(f'File {file.filename} tidak kompatibel dengan metode {method}. Format yang didukung: {", ".join(SUPPORTED_EXTENSIONS[method])}', 'error')
        return redirect(url_for('hello'))
    
    try:
        # Generate ID unik untuk proses kompresi ini
        compression_id = str(uuid.uuid4())
        compression_progress[compression_id] = {
            'status': 'starting',
            'progress': 0,
            'time_left': 'Calculating...',
            'start_time': time.time()
        }
        
        # Baca gambar yang diupload
        img = Image.open(file.stream)
        
        # Cek apakah gambar berwarna atau grayscale
        is_color_image = img.mode in ['RGB', 'RGBA']
        
        # Simpan ukuran file asli
        file.stream.seek(0, os.SEEK_END)
        file_size_before = file.stream.tell() / 1024  # Ukuran dalam KB
        file.stream.seek(0)
        
        # Periksa kompatibilitas metode dengan tipe gambar
        method_warning = None
        if is_color_image and method in ['AMBTC', 'BTC', 'SVD'] and file_size_before > 1024:  # > 1MB
            method_warning = f"Metode {method} tidak optimal untuk gambar berwarna berukuran besar. Hasil mungkin lebih besar dari file asli."
        
        # Buat nama file baru dengan timestamp untuk menghindari konflik
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"compressed_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Simpan ID kompresi dalam session untuk diakses oleh halaman result
        session['compression_id'] = compression_id
        
        # Buat thread untuk proses kompresi yang memerlukan waktu lama
        compress_thread = threading.Thread(
            target=process_compression,
            args=(img, method, quality, parameter, filepath, file_size_before, 
                  compression_id, is_color_image, file.filename)
        )
        compress_thread.daemon = True
        compress_thread.start()
        
        # Tampilkan halaman loading dengan ID kompresi untuk polling progress
        return render_template('loading.html', 
                              compression_id=compression_id,
                              method=method,
                              original_name=file.filename)
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('hello'))

def process_compression(img, method, quality, parameter, filepath, file_size_before, 
                       compression_id, is_color_image, original_filename):
    try:
        compression_progress[compression_id]['status'] = 'processing'
        
        # Terapkan metode kompresi yang dipilih
        if method == 'PIL':
            # Metode standar PIL - tidak mengubah tipe gambar
            img.save(filepath, optimize=True, quality=quality)
            compression_info = f"Quality: {quality}"
            compression_progress[compression_id]['progress'] = 100
        
        elif method == 'DCT':
            # Untuk DCT, proses gambar berwarna dengan mempertahankan warna
            if is_color_image:
                # Proses setiap channel warna secara terpisah
                img_array = np.array(img)
                compressed_channels = []
                
                # Jika RGBA, konversi ke RGB
                if img.mode == 'RGBA':
                    img_array = img_array[:,:,:3]
                    
                # Proses setiap channel warna dengan update progress
                for c in range(3):  # R, G, B channels
                    compression_progress[compression_id]['progress'] = (c * 33)
                    compression_progress[compression_id]['time_left'] = f'Processing channel {c+1}/3...'
                    channel = img_array[:,:,c]
                    compressed_channel = compress_image_dct(channel, quality, 
                                                           lambda p: update_channel_progress(compression_id, c, p))
                    compressed_channels.append(compressed_channel)
                
                # Gabungkan kembali channel
                compression_progress[compression_id]['progress'] = 90
                compression_progress[compression_id]['time_left'] = 'Finalizing...'
                compressed = np.stack(compressed_channels, axis=2).astype(np.uint8)
                cv2.imwrite(filepath, cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))  # OpenCV menggunakan BGR
            else:
                # Gambar grayscale
                img_np = np.array(img)
                compressed = compress_image_dct(img_np, quality, 
                                               lambda p: update_progress(compression_id, p))
                cv2.imwrite(filepath, compressed)
                
            compression_info = f"Quality: {quality} (DCT)"
        
        elif method == 'DWT':
            # Untuk DWT, proses gambar berwarna dengan mempertahankan warna
            level = int(parameter) if parameter and parameter.isdigit() and 1 <= int(parameter) <= 3 else 2
            
            if is_color_image:
                # Proses setiap channel warna secara terpisah
                img_array = np.array(img)
                compressed_channels = []
                
                # Jika RGBA, konversi ke RGB
                if img.mode == 'RGBA':
                    img_array = img_array[:,:,:3]
                
                # Proses setiap channel warna
                for c in range(3):  # R, G, B channels
                    compression_progress[compression_id]['progress'] = (c * 33)
                    compression_progress[compression_id]['time_left'] = f'Processing channel {c+1}/3...'
                    channel = img_array[:,:,c]
                    compressed_channel, _ = dwt_compress(channel, level=level)
                    compressed_channels.append(compressed_channel)
                    
                # Gabungkan kembali channel
                compression_progress[compression_id]['progress'] = 90
                compression_progress[compression_id]['time_left'] = 'Finalizing...'
                compressed = np.stack(compressed_channels, axis=2).astype(np.uint8)
                cv2.imwrite(filepath, cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))  # OpenCV menggunakan BGR
            else:
                # Gambar grayscale
                img_gray = img.convert('L')
                img_np = np.array(img_gray)
                
                update_progress(compression_id, 10)
                compressed, _ = dwt_compress(img_np, level=level)
                update_progress(compression_id, 90)
                
                cv2.imwrite(filepath, compressed)
                
            compression_info = f"Level: {level} (DWT)"
        
        elif method == 'BTC':
            # Konversi ke grayscale untuk metode BTC
            img_gray = img.convert('L')
            img_np = np.array(img_gray)
            
            # Parameter untuk BTC: block size (2, 4, 8)
            block_size = int(parameter) if parameter and parameter.isdigit() and int(parameter) in [2, 4, 8] else 4
            
            update_progress(compression_id, 20)
            compressed = compress_image_btc(img_np, block_size=block_size)
            update_progress(compression_id, 90)
            
            cv2.imwrite(filepath, compressed)
            compression_info = f"Block Size: {block_size} (BTC)"
            
            if is_color_image:
                method_warning = "BTC hanya bekerja dengan gambar grayscale. Gambar berwarna telah dikonversi ke grayscale."
        
        elif method == 'AMBTC':
            # Konversi ke grayscale untuk metode AMBTC
            img_gray = img.convert('L')
            img_np = np.array(img_gray)
            
            # Parameter untuk AMBTC: block size (2, 4, 8)
            block_size = int(parameter) if parameter and parameter.isdigit() and int(parameter) in [2, 4, 8] else 4
            
            update_progress(compression_id, 20)
            compressed = compress_image_ambtc(img_np, block_size=block_size)
            update_progress(compression_id, 90)
            
            cv2.imwrite(filepath, compressed)
            compression_info = f"Block Size: {block_size} (AMBTC)"
            
            if is_color_image:
                method_warning = "AMBTC hanya bekerja dengan gambar grayscale. Gambar berwarna telah dikonversi ke grayscale."
        
        elif method == 'SVD':
            # Konversi ke grayscale untuk metode SVD
            img_gray = img.convert('L')
            img_np = np.array(img_gray)
            
            update_progress(compression_id, 20)
            
            # Parameter untuk SVD: k nilai singular atau threshold energi
            if parameter and parameter.isdigit():
                compressed, energy_ratio = compress_image_svd(img_np, k=int(parameter))
                compression_info = f"Rank: {parameter}, Energy: {energy_ratio:.2%} (SVD)"
            else:
                compressed, energy_ratio = compress_image_svd(img_np)
                compression_info = f"Energy Threshold: 90%, Retained: {energy_ratio:.2%} (SVD)"
            
            update_progress(compression_id, 90)
            cv2.imwrite(filepath, compressed)
            
            if is_color_image:
                method_warning = "SVD hanya bekerja dengan gambar grayscale. Gambar berwarna telah dikonversi ke grayscale."
        
        # Hitung ukuran file setelah kompresi
        compression_progress[compression_id]['progress'] = 95
        compression_progress[compression_id]['time_left'] = 'Finalizing...'
        
        file_size_after = os.path.getsize(filepath) / 1024  # Ukuran dalam KB
        compression_ratio = (1 - (file_size_after / file_size_before)) * 100 if file_size_before > 0 else 0
        
        # Tambahkan peringatan jika hasilnya lebih besar dari aslinya
        if file_size_after > file_size_before:
            if not method_warning:
                method_warning = "Hasil kompresi lebih besar dari file asli!"
            compression_info += " ⚠️ UKURAN BERTAMBAH"
        
        # Simpan data ke database
        with app.app_context():
            # Extract the filename from the filepath
            compressed_filename = os.path.basename(filepath)
            compressed_img = CompressedImage(
                original_filename=original_filename,
                compressed_filename=compressed_filename,
                quality=quality,
                file_size_before=file_size_before,
                file_size_after=file_size_after,
                compression_ratio=compression_ratio,
                compression_method=method,
                parameter_value=str(parameter) if parameter else str(quality)
            )
            db.session.add(compressed_img)
            db.session.commit()
        
        # Format ukuran file untuk tampilan
        size_before = f"{file_size_before:.2f} KB"
        size_after = f"{file_size_after:.2f} KB"
        
        # Update status menjadi 'complete' dengan hasil
        compression_progress[compression_id]['status'] = 'complete'
        compression_progress[compression_id]['progress'] = 100
        compression_progress[compression_id]['time_left'] = 'Selesai'
        compression_progress[compression_id]['result'] = {
            'original_name': original_filename,
            'compressed_image': os.path.basename(filepath),
            'quality': quality,
            'size_before': size_before,
            'size_after': size_after,
            'compression_ratio': f"{compression_ratio:.2f}%",
            'compression_method': method,
            'compression_info': compression_info,
            'method_warning': method_warning,
            'is_color_image': is_color_image,
            'time_taken': f"{time.time() - compression_progress[compression_id]['start_time']:.2f} detik"
        }
    except Exception as e:
        compression_progress[compression_id]['status'] = 'error'
        compression_progress[compression_id]['error'] = str(e)

def update_progress(compression_id, progress):
    """Update progress untuk metode kompresi"""
    if compression_id in compression_progress:
        compression_progress[compression_id]['progress'] = progress
        # Estimasi waktu yang tersisa berdasarkan progress
        elapsed = time.time() - compression_progress[compression_id]['start_time']
        if progress > 0:
            time_left = (elapsed / progress) * (100 - progress)
            compression_progress[compression_id]['time_left'] = f"{time_left:.1f} detik"

def update_channel_progress(compression_id, channel, progress):
    """Update progress untuk channel tertentu dalam DCT"""
    if compression_id in compression_progress:
        # Setiap channel memiliki bobot 33%
        overall_progress = (channel * 33) + (progress / 3)
        compression_progress[compression_id]['progress'] = min(overall_progress, 99)
        # Estimasi waktu yang tersisa
        elapsed = time.time() - compression_progress[compression_id]['start_time']
        if overall_progress > 0:
            time_left = (elapsed / overall_progress) * (100 - overall_progress)
            compression_progress[compression_id]['time_left'] = f"{time_left:.1f} detik"

@app.route('/progress/<compression_id>')
def get_progress(compression_id):
    """Endpoint untuk polling status kompresi"""
    if compression_id not in compression_progress:
        return jsonify({'status': 'not_found'})
    
    progress_data = compression_progress[compression_id]
    
    # Jika proses sudah selesai, tambahkan data hasil kompresi
    if progress_data['status'] == 'complete' and 'result' in progress_data:
        return jsonify({
            'status': 'complete',
            'progress': 100,
            'result': progress_data['result']
        })
    elif progress_data['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': progress_data.get('error', 'Terjadi kesalahan saat kompresi')
        })
    else:
        return jsonify({
            'status': progress_data['status'],
            'progress': progress_data['progress'],
            'time_left': progress_data['time_left']
        })

@app.route('/result/<compression_id>')
def show_result(compression_id):
    """Tampilkan hasil kompresi dari ID tertentu"""
    if compression_id not in compression_progress or compression_progress[compression_id]['status'] != 'complete':
        flash('Data hasil kompresi tidak ditemukan atau belum selesai', 'error')
        return redirect(url_for('hello'))
    
    result = compression_progress[compression_id]['result']
    return render_template('result.html', **result)

@app.route('/history')
def history():
    # Tampilkan semua gambar yang sudah dikompresi
    all_compressions = CompressedImage.query.order_by(CompressedImage.compress_date.desc()).all()
    return render_template('history.html', compressions=all_compressions)

# Inisialisasi database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
