#!/usr/bin/env python3
"""Scan tracked files and the complete staged diff, never printing matched values.

Conservative heuristic audit, not a proof that secrets cannot exist.
"""
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
APPROVED = json.loads((ROOT / 'provenance/secret-scan-exceptions.json').read_text())
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,})\b'),
    'credential_assignment': re.compile(r'''(?i)(?:api_key|password|passwd|secret|access_token)\s*[=:]\s*["'][^"'\n]{6,}["']'''),
    'credential_url': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
}

def scan(text, label):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append({'file': label, 'line': number, 'kind': kind})
        approved_line = line[1:] if label == '<full-staged-diff>' and line.startswith(('+', '-')) else line
        if hashlib.sha256(approved_line.strip().encode()).hexdigest() in APPROVED:
            continue
        for match in re.finditer(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', line):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if address.is_private and not address.is_loopback and not address.is_unspecified:
                findings.append({'file': label, 'line': number, 'kind': 'private_address'})
    return findings

if __name__ == '__main__':
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    findings = []
    count = 0
    for name in filter(None, names):
        path = ROOT / name
        data = path.read_bytes()
        if len(data) > 2_000_000:
            findings.append({'file': name, 'kind': 'oversize_file'})
        if b'\0' in data:
            findings.append({'file': name, 'kind': 'binary_file'})
        findings.extend(scan(data.decode(errors='replace'), name))
        count += 1
    diff = subprocess.check_output(['git', 'diff', '--cached', '--no-ext-diff', '--binary'], cwd=ROOT).decode(errors='replace')
    findings.extend(scan(diff, '<full-staged-diff>'))
    print(json.dumps({'tracked_files_scanned': count, 'full_staged_diff_bytes': len(diff.encode()),
                      'findings': findings}, indent=2))
    raise SystemExit(bool(findings))
