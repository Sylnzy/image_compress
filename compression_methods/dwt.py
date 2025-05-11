import numpy as np
from typing import Dict, List, Tuple
import pywt
import time

def dwt2d(image: np.ndarray, level: int, wavelet: str = 'db4') -> Dict:
    """
    Menerapkan 2D Discrete Wavelet Transform pada gambar
    
    Args:
        image: Gambar grayscale 2D
        level: Level dekomposisi
        wavelet: Nama wavelet (default 'db4')
        
    Returns:
        Dict berisi subband untuk setiap level dekomposisi
    """
    result = {}
    
    # Menyimpan gambar asli
    result['original'] = image.copy()
    
    # Membuat salinan gambar untuk dekomposisi
    img_copy = image.copy().astype(np.float32)
    
    # Apply DWT untuk setiap level
    for i in range(1, level + 1):
        # Melakukan dekomposisi 2D
        coeffs = pywt.dwt2(img_copy, wavelet)
        
        # Mendapatkan subband (LL, (LH, HL, HH))
        LL, (LH, HL, HH) = coeffs
        
        # Threshold koefisien detail
        T = 10  # Nilai threshold
        LH = pywt.threshold(LH, T, 'soft')
        HL = pywt.threshold(HL, T, 'soft')
        HH = pywt.threshold(HH, T, 'soft')
        
        # Menyimpan subband untuk level ini
        result[f'level_{i}'] = {
            'LL': LL,
            'LH': LH,
            'HL': HL,
            'HH': HH
        }
        
        # Meneruskan subband LL untuk dekomposisi level berikutnya
        img_copy = LL
        
    return result

def dwt_compress(image: np.ndarray, level: int = 2, wavelet: str = 'db4') -> Tuple[np.ndarray, Dict]:
    """
    Kompresi gambar menggunakan DWT
    
    Args:
        image: Gambar grayscale
        level: Level dekomposisi (default 2)
        wavelet: Nama wavelet (default 'db4')
        
    Returns:
        Tuple dari (gambar_terkompresi, koefisien_dwt)
    """
    # Terapkan DWT
    coeffs = dwt2d(image, level, wavelet)
    
    # Rekonstruksi gambar dari koefisien yang sudah di-threshold
    reconstructed = reconstructed_image_from_dwt(coeffs, level, wavelet)
    
    # Clip nilai ke range [0, 255] dan konversi ke uint8
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed, coeffs

def reconstructed_image_from_dwt(coeffs: Dict, level: int, wavelet: str) -> np.ndarray:
    """
    Merekonstruksi gambar dari koefisien DWT
    
    Args:
        coeffs: Dictionary koefisien dari dwt2d
        level: Level dekomposisi
        wavelet: Nama wavelet
        
    Returns:
        Gambar hasil rekonstruksi
    """
    # Mulai dari level terdalam
    current_level = level
    
    # Ambil approximation dari level terdalam
    approx = coeffs[f'level_{current_level}']['LL']
    
    # Rekonstruksi dari dalam ke luar
    while current_level > 0:
        # Ambil detail coefficients dari level saat ini
        LH = coeffs[f'level_{current_level}']['LH']
        HL = coeffs[f'level_{current_level}']['HL']
        HH = coeffs[f'level_{current_level}']['HH']
        
        # Rekonstruksi satu level
        approx = pywt.idwt2((approx, (LH, HL, HH)), wavelet)
        
        current_level -= 1
        
    return approx

def dwt_compress(img, level=2, wavelet='haar', threshold_method='hard', progress_callback=None):
    """
    Kompresi gambar menggunakan Discrete Wavelet Transform (DWT)
    dengan callback progress.
    
    Args:
        img: Gambar input dalam format numpy array
        level: Level dekomposisi wavelet (1-3)
        wavelet: Tipe wavelet yang digunakan (default: 'haar')
        threshold_method: Metode thresholding ('hard' atau 'soft')
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
    
    Returns:
        Tuple (compressed_img, compression_info)
    """
    # Report starting progress
    if progress_callback:
        progress_callback(5)
    
    # Convert to float for DWT
    img_float = img.astype(float)
    
    # Lakukan dekomposisi wavelet multilevel
    start_time = time.time()
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)
    
    if progress_callback:
        progress_callback(30)  # Dekomposisi selesai
    
    # Tentukan threshold
    # Semakin tinggi nilai threshold, semakin banyak koefisien yang di-nol-kan
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Tentukan nilai threshold berdasarkan persentase energi yang ingin dipertahankan
    abs_coeff = np.abs(coeff_arr)
    threshold = np.percentile(abs_coeff[abs_coeff > 0], 40)  # Nilai percentile untuk filtering
    
    if progress_callback:
        progress_callback(40)  # Threshold ditentukan
    
    # Aplikasikan thresholding pada koefisien
    if threshold_method == 'hard':
        coeff_arr[abs_coeff < threshold] = 0  # Hard thresholding
    else:  # soft
        coeff_arr = pywt.threshold(coeff_arr, threshold, mode='soft')  # Soft thresholding
    
    if progress_callback:
        progress_callback(60)  # Thresholding selesai
    
    # Rekonstruksi koefisien menjadi array 2D
    compressed_coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    
    # Rekonstruksi gambar dari koefisien yang dimodifikasi
    compressed = pywt.waverec2(compressed_coeffs, wavelet)
    
    if progress_callback:
        progress_callback(90)  # Rekonstruksi selesai
    
    # Clip nilai-nilai di luar range [0, 255]
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    
    # Hitung persentase koefisien nol (menunjukkan kompresi)
    zero_percent = 100 * np.sum(coeff_arr == 0) / coeff_arr.size
    compression_info = f"Wavelet: {wavelet}, Level: {level}, Zero coefficients: {zero_percent:.1f}%"
    
    if progress_callback:
        progress_callback(100)  # Proses selesai
    
    return compressed, compression_info