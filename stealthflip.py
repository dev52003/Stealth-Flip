import argparse
import sys
import os
import random
from PIL import Image
import numpy as np
import zlib
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import requests
import time  # for rate limiting

def neg_trans(img):
    inverted_data = [(255 - r, 255 - g, 255 - b, a) for r, g, b, a in img.getdata()]
    img.putdata(inverted_data)

def compute_seed_from_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width + height

def generate_key(partial_phrase, user_input):
    key_material = partial_phrase + user_input
    key_material = key_material.ljust(32, '\0')  # Ensure the key is exactly 32 bytes (256 bits) for AES-256
    return key_material.encode()

def encrypt_data(data, key):
    backend = default_backend()
    iv = os.urandom(16)  # Generate a random IV
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return iv + encrypted_data

def decrypt_data(data, key):
    backend = default_backend()
    iv = data[:16]  # Extract IV from the data
    encrypted_data = data[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=backend)
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data

def hide_file_in_png(image_path, file_to_hide, output_image_path):
    # Retrieve partial phrase from the server with rate limiting and limited attempts
    partial_phrase = 'abcdefgh' 

    # Check if retrieval was successful
    if partial_phrase is None:
        return

    # Get user input for the 4 characters
    user_input = input("Enter 4 characters for encryption: ").strip()
    if len(user_input) != 4:
        raise ValueError("Please enter exactly 4 characters for encryption")

    # Generate encryption key
    key = generate_key(partial_phrase, user_input)

    # Image processing...
    seed = compute_seed_from_image_dimensions(image_path)
    prng = random.Random(seed)

    img = Image.open(image_path)
    
    if img.mode not in ['RGB', 'RGBA', 'P', 'L']:
        raise ValueError("Image mode must be RGB, RGBA, P (palette-based), or L (grayscale).")

    if img.mode == 'P' or img.mode == 'L':
        img = img.convert('RGB')


    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Applying negative transform on the input image
    neg_trans(img)
    
    host_format = img.format  
    if host_format is None:
        file_extension = os.path.splitext(image_path)[1].lower()
        extension_to_format = {
            '.tga': 'TGA',
            '.png': 'PNG',
            '.bmp': 'BMP',
            '.tif': 'TIFF',
            '.tiff': 'TIFF',
        }
        host_format = extension_to_format.get(file_extension)

    supported_formats = {'TGA', 'TIFF', 'BMP', 'PNG'}
    if host_format not in supported_formats:
        raise ValueError(f"Unsupported image format: {host_format}")
        
    pixels = np.array(img)
    
    with open(file_to_hide, 'rb') as f:
        file_bytes = f.read()
    
    compressed_data = zlib.compress(file_bytes)

    encrypted_data = encrypt_data(compressed_data,key)
    
    filename = os.path.basename(file_to_hide).encode()
    filename_size = len(filename)

    data_to_encode = (filename_size.to_bytes(4, 'big') + filename + encrypted_data)
    
    file_size = len(data_to_encode)
    num_pixels_required = file_size * 8  
    if num_pixels_required > pixels.size // 4:
        raise ValueError("Image is not large enough to hide the file.")

    pixel_indices = list(range(pixels.size // 4))
    prng.shuffle(pixel_indices)  

    for i in range(64):
        idx = pixel_indices[i]
        bit = (file_size >> (63 - i)) & 0x1
        if (pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] & 0x1) != bit:
            pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] ^= 0x1

    for i, byte in enumerate(data_to_encode):
        for bit in range(8):
            idx = pixel_indices[64 + i * 8 + bit]
            if (pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] & 0x1) != ((byte >> (7 - bit)) & 0x1):
                pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] ^= 0x1

    if os.path.exists(output_image_path):
        overwrite = input(f"The file '{output_image_path}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Extraction cancelled.")
            return
    
    new_img = Image.fromarray(pixels, 'RGBA')

    # Applying negative transfomrm on the encoded (negative) output
    neg_trans(new_img)

    if host_format == 'PNG':
        new_img.save(output_image_path, format='PNG', optimize=True)
    elif host_format == 'BMP':
        new_img.save(output_image_path, format='BMP', optimize=True)
    elif host_format == 'TGA':
        new_img.save(output_image_path, format='TGA', optimize=True)
    elif host_format == 'TIFF':
        new_img.save(output_image_path, format='TIFF', optimize=True)
    else:
        raise ValueError(f"Unsupported image format: {host_format}")

    print(f"File '{file_to_hide}' has been successfully hidden in '{output_image_path}'.")

def extract_file_from_png(image_path, output_file_path):
    # Retrieve partial phrase from the server with rate limiting and limited attempts
    partial_phrase = 'abcdefgh'

    # Check if retrieval was successful
    if partial_phrase is None:
        return

    # Get user input for the 4 characters
    user_input = input("Enter 4 characters for decryption: ").strip()
    if len(user_input) != 4:
        raise ValueError("Please enter exactly 4 characters for decryption")

    # Generate decryption key
    key = generate_key(partial_phrase, user_input)

    # Image processing...
    seed = compute_seed_from_image_dimensions(image_path)
    prng = random.Random(seed)

    img = Image.open(image_path)

    if img.mode not in ['RGB', 'RGBA']:
        raise ValueError("Image must be in RGB or RGBA format.")
    
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Applying negative transform on output
    neg_trans(img)

    pixels = np.array(img)
    
    flat_pixels = pixels.flatten()
    
    channel_multiplier = 4

    file_size = 0
    for i in range(64):
        file_size = (file_size << 1) | (flat_pixels[i * channel_multiplier] & 0x1)
    
    num_bytes_to_extract = file_size
    
    extracted_bytes = []

    pixel_indices = list(range(pixels.size // 4))
    prng.shuffle(pixel_indices)  

    file_size = 0
    for i in range(64):
        idx = pixel_indices[i]
        file_size = (file_size << 1) | (pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] & 0x1)

    num_bytes_to_extract = file_size

    extracted_bytes = []
    for i in range(num_bytes_to_extract):
        byte = 0
        for bit in range(8):
            idx = pixel_indices[64 + i * 8 + bit]
            byte = (byte << 1) | (pixels[idx // pixels.shape[1], idx % pixels.shape[1], 0] & 0x1)
        extracted_bytes.append(byte)
    
    data_to_decode = bytes(extracted_bytes)

    filename_size = int.from_bytes(data_to_decode[:4], 'big')
    filename = data_to_decode[4:4 + filename_size].decode()
    
    offset = 4 + filename_size
    encrypted_data = data_to_decode[offset:]

    try:
        decrypted_data = decrypt_data(encrypted_data, key)
    except ValueError as e:
        print("Enter correct passphrase !")
        return

    decompressed_data = zlib.decompress(decrypted_data)

    if not output_file_path:
        output_file_path = os.path.join(os.getcwd(), filename)

    if os.path.exists(output_file_path):
        overwrite = input(f"The file '{output_file_path}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Extraction cancelled.")
            return
        
    with open(output_file_path, 'wb') as f:
        f.write(decompressed_data)

    print(f"File extracted to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description='SecretPixel - Advanced Steganography Tool', epilog="Example commands:\n"
                                            "  Hide: python secret_pixel.py hide host.png secret.txt output.png\n"
                                            "  Extract: python secret_pixel.py extract carrier.png [extracted.txt]",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')

    hide_parser = subparsers.add_parser('hide', help='Hide a file inside an image', epilog="Example: python secret_pixel.py hide host.png secret.txt output.png", formatter_class=argparse.RawDescriptionHelpFormatter)
    hide_parser.add_argument('host', type=str, help='Path to the host image')
    hide_parser.add_argument('secret', type=str, help='Path to the secret file to hide')
    hide_parser.add_argument('output', type=str, help='Path to the output image with embedded data')

    extract_parser = subparsers.add_parser('extract', help='Extract a file from an image', epilog="Example: python secret_pixel.py extract carrier.png  [extracted.txt]",
                                           formatter_class=argparse.RawDescriptionHelpFormatter)
    extract_parser.add_argument('carrier', type=str, help='Path to the image with embedded data')
    extract_parser.add_argument('extracted', nargs='?', type=str, default=None, help='Path to save the extracted secret file (optional, defaults to the original filename)')


    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == 'hide':
        hide_file_in_png(args.host, args.secret, args.output)
    elif args.command == 'extract':
        extract_file_from_png(args.carrier, args.extracted)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()