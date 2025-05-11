import numpy as np
from typing import Tuple
import math
import cv2
import time

def ambtc(block: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Implementasi AMBTC (Absolute Moment Block Truncation Coding)
    
    Args:
        block: Blok piksel grayscale
        
    Returns:
        Tuple dari (bitplane, xH, xL)
    """
    # Hitung mean dan standard deviation
    mu = np.mean(block)
    sigma = np.std(block)
    
    # Buat bitplane
    bitplane = (block >= mu).astype(np.uint8)
    
    # Hitung proporsi piksel
    p1 = np.mean(bitplane)
    p0 = 1 - p1
    
    # Hindari pembagian dengan nol
    if p1 > 0 and p0 > 0:
        xH = mu + sigma * math.sqrt(p0/p1)
        xL = mu - sigma * math.sqrt(p1/p0)
    else:
        xH = 255 if p1 > 0 else mu
        xL = 0 if p0 > 0 else mu
    
    return bitplane, float(xH), float(xL)

def ambtc_decompress(bitplane: np.ndarray, xH: float, xL: float) -> np.ndarray:
    """
    Dekompresi gambar yang dikompresi dengan AMBTC
    
    Args:
        bitplane: Bitplane hasil kompresi
        xH: Nilai rekonstruksi untuk bit 1
        xL: Nilai rekonstruksi untuk bit 0
        
    Returns:
        Blok gambar hasil dekompresi
    """
    # Rekonstruksi gambar
    reconstructed = np.zeros_like(bitplane, dtype=np.float32)
    reconstructed[bitplane == 1] = xH
    reconstructed[bitplane == 0] = xL
    
    # Clip nilai ke range [0, 255]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed

def compress_image_ambtc(image: np.ndarray, block_size: int = 4) -> np.ndarray:
    """
    Kompresi gambar menggunakan AMBTC
    
    Args:
        image: Gambar grayscale
        block_size: Ukuran blok untuk pemrosesan (default 4x4)
        
    Returns:
        Gambar hasil kompresi
    """
    h, w = image.shape
    
    # Pad image jika perlu agar sesuai dengan block_size
    padded_h = h if h % block_size == 0 else h + (block_size - h % block_size)
    padded_w = w if w % block_size == 0 else w + (block_size - w % block_size)
    
    padded_image = np.zeros((padded_h, padded_w), dtype=np.uint8)
    padded_image[:h, :w] = image
    
    # Gambar hasil
    result = np.zeros_like(padded_image)
    
    # Proses setiap blok
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            # Ekstrak blok
            block = padded_image[i:i+block_size, j:j+block_size]
            
            # Kompresi blok dengan AMBTC
            bitplane, xH, xL = ambtc(block)
            
            # Dekompresi blok
            decompressed = ambtc_decompress(bitplane, xH, xL)
            
            # Masukkan hasil ke gambar akhir
            result[i:i+block_size, j:j+block_size] = decompressed
    
    # Kembalikan gambar dengan ukuran asli
    return result[:h, :w]

def ambtc_compress(img, block_size=4, progress_callback=None):
    """
    Kompresi gambar menggunakan Absolute Moment Block Truncation Coding (AMBTC)
    dengan callback progress.
    
    Args:
        img: Gambar input dalam format numpy array
        block_size: Ukuran blok (2, 4, atau 8)
        progress_callback: Fungsi callback untuk melaporkan progress (0-100)
    
    Returns:
        Tuple (compressed_img, compression_info)
    """
    # Pastikan gambar dalam format grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if progress_callback:
        progress_callback(5)
    
    start_time = time.time()
    
    h, w = img.shape
    
    # Padding gambar jika diperlukan
    pad_h = 0 if h % block_size == 0 else block_size - (h % block_size)
    pad_w = 0 if w % block_size == 0 else block_size - (w % block_size)
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), 'constant')
        h, w = img.shape
    
    # Inisialisasi gambar hasil
    compressed = np.zeros((h, w), dtype=np.uint8)
    
    # Hitung total blok untuk progress tracking
    total_blocks = (h // block_size) * (w // block_size)
    processed_blocks = 0
    
    if progress_callback:
        progress_callback(10)
    
    # Proses setiap blok
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Ambil blok dari gambar
            block = img[i:i+block_size, j:j+block_size]
            
            # Hitung rata-rata blok
            block_mean = np.mean(block)
            
            # Pisahkan pixel di atas dan di bawah rata-rata
            mask_high = block >= block_mean
            mask_low = ~mask_high
            
            # Hitung jumlah pixel di atas dan di bawah rata-rata
            num_high = np.sum(mask_high)
            num_low = block_size * block_size - num_high
            
            # Rekonstruksi nilai untuk a (rendah) dan b (tinggi)
            if num_high > 0:
                high_val = np.sum(block[mask_high]) / num_high
            else:
                high_val = block_mean
                
            if num_low > 0:
                low_val = np.sum(block[mask_low]) / num_low
            else:
                low_val = block_mean
            
            # Rekonstruksi blok
            block_compressed = np.zeros_like(block)
            block_compressed[mask_high] = high_val
            block_compressed[mask_low] = low_val
            
            # Simpan blok yang dikompresi ke gambar hasil
            compressed[i:i+block_size, j:j+block_size] = block_compressed
            
            # Update progress
            processed_blocks += 1
            if progress_callback and processed_blocks % 100 == 0:
                progress = 10 + (processed_blocks / total_blocks) * 85
                progress_callback(progress)
    
    # Potong padding jika ada
    if pad_h > 0 or pad_w > 0:
        compressed = compressed[:h-pad_h, :w-pad_w]
    
    compression_info = f"Block size: {block_size}x{block_size}"
    
    if progress_callback:
        progress_callback(100)
    
    return compressed, compression_info