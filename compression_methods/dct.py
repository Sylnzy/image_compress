import numpy as np
import cv2
import math
from scipy.fftpack import dct, idct
from typing import List
import time

def compress_block_dct(block: List[List[float]], Qtable: List[List[int]]) -> List[int]:
    """
    Kompresi blok gambar 8x8 menggunakan DCT
    
    Args:
        block: Matriks 8x8 pixel grayscale
        Qtable: Tabel kuantisasi JPEG standar
        
    Returns:
        List koefisien terkuantisasi dalam urutan zig-zag
    """
    N = 8  # Ukuran blok
    
    # Konversi ke numpy array untuk mempermudah perhitungan
    block_np = np.array(block, dtype=float)
    Qtable_np = np.array(Qtable, dtype=int)
    
    # Menginisialisasi matriks output DCT
    dct_block = np.zeros((N, N), dtype=float)
    
    # Forward 2D-DCT
    for u in range(N):
        for v in range(N):
            # Menghitung koefisien C(u) dan C(v)
            cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
            cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
            
            # Menghitung DCT
            sum_val = 0.0
            for x in range(N):
                for y in range(N):
                    sum_val += block_np[x, y] * \
                              math.cos((2 * x + 1) * u * math.pi / (2 * N)) * \
                              math.cos((2 * y + 1) * v * math.pi / (2 * N))
            
            dct_block[u, v] = (2.0 / N) * cu * cv * sum_val
    
    # Kuantisasi
    quantized = np.round(dct_block / Qtable_np)
    
    # Zigzag scan untuk mengembalikan array 1D
    zigzag_order = get_zigzag_order(N)
    zigzag_coeffs = [int(quantized[i, j]) for i, j in zigzag_order]
    
    return zigzag_coeffs

def get_zigzag_order(n: int) -> List[tuple]:
    """
    Mengembalikan urutan indeks untuk zigzag scan matriks n x n
    """
    result = []
    for i in range(n * 2 - 1):
        if i % 2 == 0:  # Genap - arah naik
            for j in range(min(i, n - 1), max(0, i - n + 1) - 1, -1):
                result.append((j, i - j))
        else:  # Ganjil - arah turun
            for j in range(max(0, i - n + 1), min(i + 1, n)):
                result.append((j, i - j))
    return result

def dct_decompress(zigzag_coeffs: List[int], Qtable: List[List[int]], block_size: int = 8) -> np.ndarray:
    """
    Dekompresi koefisien DCT kembali ke blok gambar
    
    Args:
        zigzag_coeffs: List koefisien DCT dalam urutan zig-zag
        Qtable: Tabel kuantisasi
        block_size: Ukuran blok (default 8)
        
    Returns:
        Blok gambar 8x8 hasil dekompresi
    """
    N = block_size
    Qtable_np = np.array(Qtable, dtype=int)
    
    # Reverse zigzag scan
    zigzag_order = get_zigzag_order(N)
    quantized = np.zeros((N, N), dtype=float)
    
    for (i, j), coeff in zip(zigzag_order, zigzag_coeffs):
        quantized[i, j] = coeff
    
    # Inverse quantization
    dequantized = quantized * Qtable_np
    
    # Inverse DCT
    idct_block = np.zeros((N, N), dtype=float)
    
    for x in range(N):
        for y in range(N):
            sum_val = 0.0
            for u in range(N):
                for v in range(N):
                    # Menghitung koefisien C(u) dan C(v)
                    cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
                    cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
                    
                    sum_val += cu * cv * dequantized[u, v] * \
                              math.cos((2 * x + 1) * u * math.pi / (2 * N)) * \
                              math.cos((2 * y + 1) * v * math.pi / (2 * N))
            
            idct_block[x, y] = (2.0 / N) * sum_val
    
    # Clip values to valid range [0, 255]
    idct_block = np.clip(idct_block, 0, 255).astype(np.uint8)
    
    return idct_block

def compress_image_dct(img, quality, progress_callback=None):
    """
    Kompresi gambar menggunakan Discrete Cosine Transform (DCT)
    dengan parameter kualitas dan callback progress.
    
    Args:
        img: Gambar input dalam format numpy array
        quality: Parameter kualitas (1-100)
        progress_callback: Callback function untuk melaporkan progress (0-100)
    
    Returns:
        Gambar hasil kompresi
    """
    # Simpan dimensi asli
    height, width = img.shape[:2]
    
    # Konversi ke float untuk kalkulasi DCT
    img_float = np.float32(img)
    
    # Pastikan ukuran gambar bisa dibagi dengan 8
    pad_h, pad_w = 0, 0
    if height % 8 != 0 or width % 8 != 0:
        # Pad gambar
        pad_h = 8 - (height % 8) if height % 8 != 0 else 0
        pad_w = 8 - (width % 8) if width % 8 != 0 else 0
        img_float = np.pad(img_float, ((0, pad_h), (0, pad_w)), 'constant')
    
    # Dapatkan dimensi baru
    height, width = img_float.shape[:2]
    
    # Jumlah total blok 8x8
    total_blocks = (height // 8) * (width // 8)
    processed_blocks = 0
    
    # Inisialisasi array hasil
    result = np.zeros_like(img_float)
    
    # Konversi parameter kualitas menjadi thresholding value
    # quality 1 = threshold tinggi (kompresi tinggi), quality 100 = threshold rendah (kualitas tinggi)
    threshold = 0.1 + (1.0 - (quality / 100.0)) * 0.9  # 0.1 (high quality) to 1.0 (low quality)
    
    # Proses per blok 8x8
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Potong blok 8x8
            block = img_float[i:i+8, j:j+8]
            
            # Aplikasikan DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Thresholding (zeroing) koefisien kecil untuk kompresi
            # Semakin tinggi threshold, semakin banyak koefisien yang di-zero-kan
            thresh = threshold * np.max(np.abs(dct_block))
            dct_block[np.abs(dct_block) < thresh] = 0
            
            # Aplikasikan inverse DCT untuk mendapatkan blok gambar kembali
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            
            # Masukkan kembali ke gambar hasil
            result[i:i+8, j:j+8] = idct_block
            
            # Update progress
            processed_blocks += 1
            if progress_callback and processed_blocks % 100 == 0:  # Update setiap 100 blok untuk efisiensi
                progress = (processed_blocks / total_blocks) * 100
                progress_callback(progress)
    
    # Potong kembali ke ukuran asli jika sebelumnya dipadding
    result = result[:height-pad_h if pad_h else height, :width-pad_w if pad_w else width]
    
    # Clip nilai dan konversi kembali ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Lapor progress 100% di akhir
    if progress_callback:
        progress_callback(100)
    
    return result

def dct_compress(image, block_size=8, quality_factor=50, progress_callback=None):
    """
    Kompresi gambar menggunakan Discrete Cosine Transform (DCT)
    dengan callback progress.
    
    Args:
        image: Gambar input dalam format numpy array
        block_size: Ukuran blok DCT (default 8)
        quality_factor: Faktor kualitas 1-100 (semakin tinggi, semakin baik kualitasnya)
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
        
    Returns:
        Gambar terkompresi
    """
    if progress_callback:
        progress_callback(5)
    
    start_time = time.time()
    
    # Pastikan gambar adalah grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Dapatkan dimensi gambar
    height, width = image.shape
    
    # Padding gambar jika diperlukan
    pad_h = 0 if height % block_size == 0 else block_size - (height % block_size)
    pad_w = 0 if width % block_size == 0 else block_size - (width % block_size)
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), 'constant')
        height, width = image.shape
    
    # Buat matriks kuantisasi
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Sesuaikan matriks kuantisasi berdasarkan faktor kualitas
    if quality_factor < 50:
        scale_factor = 5000 / quality_factor
    else:
        scale_factor = 200 - 2 * quality_factor
    
    quantization_matrix = np.floor((quantization_matrix * scale_factor + 50) / 100)
    quantization_matrix = np.clip(quantization_matrix, 1, 255).astype(np.uint8)
    
    # Jika block_size bukan 8, sesuaikan matriks kuantisasi
    if block_size != 8:
        # Sederhananya, kita gunakan rata-rata untuk block_size yang berbeda
        q_mean = np.mean(quantization_matrix)
        quantization_matrix = np.ones((block_size, block_size)) * q_mean
    
    # Inisialisasi hasil
    result = np.zeros_like(image, dtype=np.float32)
    
    # Hitung total blok untuk progress tracking
    total_blocks = (height // block_size) * (width // block_size)
    processed_blocks = 0
    
    if progress_callback:
        progress_callback(10)
    
    # Proses setiap blok
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Ambil blok
            block = image[y:y+block_size, x:x+block_size].astype(np.float32) - 128.0
            
            # Lakukan DCT
            dct_block = cv2.dct(block)
            
            # Kuantisasi
            if block_size == 8:
                quantized = np.round(dct_block / quantization_matrix)
            else:
                # Untuk block_size yang berbeda, kita akan menyesuaikan kuantisasi
                quantized = np.round(dct_block / quantization_matrix)
            
            # Dequantisasi
            dequantized = quantized * quantization_matrix
            
            # Lakukan IDCT
            idct_block = cv2.idct(dequantized)
            
            # Simpan ke hasil
            result[y:y+block_size, x:x+block_size] = idct_block + 128.0
            
            # Update progress
            processed_blocks += 1
            if progress_callback and processed_blocks % 100 == 0:
                progress = 10 + (processed_blocks / total_blocks) * 85
                progress_callback(progress)
    
    # Clip nilai ke [0, 255] dan konversi ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Potong padding jika ada
    if pad_h > 0 or pad_w > 0:
        result = result[:height-pad_h, :width-pad_w]
    
    if progress_callback:
        progress_callback(100)
    
    return result