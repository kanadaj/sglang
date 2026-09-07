#!/usr/bin/env python3
"""Synthetic PNG + OpenAI image_url request; --send opts into HTTP."""
import argparse
import base64
import json
import os
import struct
import urllib.request
import zlib

def image():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = []
    for y in range(256):
        rows.append(b'\0' + b''.join(b'\xff\0\0' if 64 <= x < 192 and 64 <= y < 192 else b'\xff\xff\xff' for x in range(256)))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b'')

def payload():
    return {'model': os.environ.get('MODEL_NAME', 'qwen-vision'), 'temperature': 0, 'max_tokens': 128,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe the shape and color. Answer briefly.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image()).decode()}}]}]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--send', action='store_true')
    args = parser.parse_args()
    data = json.dumps(payload()).encode()
    if not args.send:
        print(data.decode())
    else:
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('OPENAI_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
        url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:30000/v1').rstrip('/') + '/chat/completions'
        request = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=180) as response:
            print(response.read().decode())
