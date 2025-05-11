import numpy as np
import cv2
import time
from typing import Tuple

def svd_compress(image, k, progress_callback=None):
    """
    Kompresi gambar menggunakan Singular Value Decomposition (SVD).
    
    Args:
        image: Gambar input dalam format numpy array
        k: Jumlah nilai singular yang dipertahankan (nilai rank)
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
    
    Returns:
        Gambar terkompresi
    """
    if progress_callback:
        progress_callback(5)  # Mulai progress
    
    start_time = time.time()
    
    # Cek gambar grayscale atau RGB
    is_rgb = len(image.shape) == 3 and image.shape[2] == 3
    
    if is_rgb:
        # Proses secara terpisah untuk setiap channel
        result = np.zeros_like(image, dtype=np.uint8)
        channels = cv2.split(image)
        
        for i, channel in enumerate(channels):
            if progress_callback:
                progress_value = 10 + i * 30
                progress_callback(progress_value)
            
            # SVD decomposition
            U, sigma, Vt = np.linalg.svd(channel, full_matrices=False)
            
            # Truncate matrices untuk kompresi
            U_k = U[:, :k]
            sigma_k = np.diag(sigma[:k])
            Vt_k = Vt[:k, :]
            
            # Rekonstruksi gambar
            compressed = np.dot(U_k, np.dot(sigma_k, Vt_k))
            
            # Clip nilai dan konversi ke uint8
            compressed = np.clip(compressed, 0, 255).astype(np.uint8)
            result[:, :, i] = compressed
    else:
        if progress_callback:
            progress_callback(10)
        
        # SVD decomposition untuk grayscale
        U, sigma, Vt = np.linalg.svd(image, full_matrices=False)
        
        if progress_callback:
            progress_callback(50)
        
        # Truncate matrices untuk kompresi
        U_k = U[:, :k]
        sigma_k = np.diag(sigma[:k])
        Vt_k = Vt[:k, :]
        
        # Rekonstruksi gambar
        result = np.dot(U_k, np.dot(sigma_k, Vt_k))
        
        # Clip nilai dan konversi ke uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    if progress_callback:
        progress_callback(100)  # Selesai
    
    return result

def compress_image_svd(image: np.ndarray, k: int = None, energy_threshold: float = 0.9, progress_callback=None) -> Tuple[np.ndarray, float]:
    """
    Kompresi gambar menggunakan SVD
    
    Args:
        image: Gambar grayscale
        k: Jumlah nilai singular (rank) yang dipertahankan. Jika None, akan dihitung berdasarkan energy_threshold.
        energy_threshold: Rasio energi minimum yang dipertahankan (0.0-1.0), default 0.9 (90%)
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
        
    Returns:
        Tuple dari (gambar_terkompresi, rasio_energi)
    """
    # Konversi ke float untuk perhitungan SVD
    img_float = image.astype(np.float32)
    
    # Jika k tidak diberikan, hitung berdasarkan energy_threshold
    if k is None:
        # Hitung SVD untuk mendapatkan nilai singular
        _, S, _ = np.linalg.svd(img_float, full_matrices=False)
        
        # Hitung rasio energi kumulatif
        energy_total = np.sum(S**2)
        energy_cumsum = np.cumsum(S**2)
        energy_ratio = energy_cumsum / energy_total
        
        # Tentukan k berdasarkan threshold
        k = np.argmax(energy_ratio >= energy_threshold) + 1
    
    # Kompresi menggunakan SVD
    img_compressed = svd_compress(img_float, k, progress_callback)
    
    # Clip nilai dan konversi kembali ke uint8
    img_compressed = np.clip(img_compressed, 0, 255).astype(np.uint8)
    
    return img_compressed, energy_ratio