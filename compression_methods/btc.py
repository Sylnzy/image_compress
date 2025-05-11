import numpy as np
import cv2
import time
from typing import Tuple, Dict, List

def btcCompress(block: np.ndarray) -> Dict:
    """
    Implementasi BTC (Block Truncation Coding) untuk kompresi gambar
    
    Args:
        block: Blok mxn pixel grayscale
        
    Returns:
        Dictionary berisi bitplane dan nilai rekonstruksi
    """
    # Hitung rata-rata blok
    mu = np.mean(block)
    
    # Bentuk bitplane: 1 jika pixel >= mu, 0 jika tidak
    bitplane = (block >= mu).astype(np.uint8)
    
    # Hitung jumlah bit 1 dan 0
    count1 = np.sum(bitplane)
    count0 = block.size - count1
    
    # Hitung nilai rekonstruksi
    if count1 > 0:
        xH = np.sum(block[bitplane == 1]) / count1
    else:
        xH = mu  # Hindari pembagian dengan nol
        
    if count0 > 0:
        xL = np.sum(block[bitplane == 0]) / count0
    else:
        xL = mu  # Hindari pembagian dengan nol
    
    return {
        "bitplane": bitplane,
        "xH": float(xH),
        "xL": float(xL)
    }

def btc_decompress(compressed_data: Dict) -> np.ndarray:
    """
    Dekompresi gambar yang dikompresi dengan BTC
    
    Args:
        compressed_data: Dictionary hasil kompresi BTC
        
    Returns:
        Blok gambar hasil dekompresi
    """
    bitplane = compressed_data["bitplane"]
    xH = compressed_data["xH"]
    xL = compressed_data["xL"]
    
    # Rekonstruksi gambar
    reconstructed = np.zeros_like(bitplane, dtype=np.float32)
    reconstructed[bitplane == 1] = xH
    reconstructed[bitplane == 0] = xL
    
    return reconstructed.astype(np.uint8)

def get_block_mean(block):
    """Menghitung rata-rata nilai dalam blok"""
    return np.mean(block)

def get_block_std(block):
    """Menghitung standar deviasi nilai dalam blok"""
    return np.std(block)

def compress_image_btc(image, block_size=4, progress_callback=None):
    """
    Kompresi gambar menggunakan Block Truncation Coding (BTC)
    dengan callback progress.
    
    Args:
        image: Gambar grayscale dalam format numpy array
        block_size: Ukuran blok (2, 4, atau 8)
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
    
    Returns:
        Gambar hasil kompresi
    """
    if progress_callback:
        progress_callback(5)
    
    start_time = time.time()
    
    # Pastikan gambar adalah grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    height, width = image.shape
    
    # Padding gambar jika diperlukan
    pad_h = 0 if height % block_size == 0 else block_size - (height % block_size)
    pad_w = 0 if width % block_size == 0 else block_size - (width % block_size)
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), 'constant')
        height, width = image.shape
    
    # Inisialisasi hasil kompresi
    result = np.zeros_like(image)
    
    # Hitung total blok untuk progress tracking
    total_blocks = (height // block_size) * (width // block_size)
    processed_blocks = 0
    
    if progress_callback:
        progress_callback(10)
    
    # Proses setiap blok
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Ambil blok yang akan diproses
            block = image[y:y+block_size, x:x+block_size]
            
            # Hitung rata-rata dan standar deviasi blok
            block_mean = get_block_mean(block)
            block_std = get_block_std(block)
            
            # Buat bit map (1 jika pixel > mean, 0 jika sebaliknya)
            bit_map = block > block_mean
            
            # Hitung nilai rekonstruksi untuk pixel yang di-encode
            if np.sum(bit_map) == 0:  # Semua pixel di bawah mean
                a = block_mean
                b = block_mean
            elif np.sum(bit_map) == block_size * block_size:  # Semua pixel di atas mean
                a = block_mean
                b = block_mean
            else:
                # Hitung nilai rekonstruksi a dan b
                n = block_size * block_size
                q = np.sum(bit_map)  # Jumlah pixel di atas mean
                
                # Rumus BTC
                a = block_mean - block_std * np.sqrt(q / (n - q))
                b = block_mean + block_std * np.sqrt((n - q) / q)
            
            # Rekonstruksi blok
            reconstructed_block = np.full((block_size, block_size), a)
            reconstructed_block[bit_map] = b
            
            # Simpan hasil ke gambar akhir
            result[y:y+block_size, x:x+block_size] = reconstructed_block
            
            # Update progress
            processed_blocks += 1
            if progress_callback and processed_blocks % 100 == 0:
                progress = 10 + (processed_blocks / total_blocks) * 85
                progress_callback(progress)
    
    # Potong hasil jika sebelumnya dipadding
    if pad_h > 0 or pad_w > 0:
        result = result[:height-pad_h, :width-pad_w]
    
    if progress_callback:
        progress_callback(100)
    
    return result