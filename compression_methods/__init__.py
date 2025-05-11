from compression_methods.dct import compress_image_dct
from compression_methods.dwt import dwt_compress
from compression_methods.btc import compress_image_btc
from compression_methods.ambtc import compress_image_ambtc
from compression_methods.svd import compress_image_svd

__all__ = ['compress_image_dct', 'dwt_compress', 'compress_image_btc', 'compress_image_ambtc', 'compress_image_svd']