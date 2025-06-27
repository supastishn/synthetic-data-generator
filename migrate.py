#!/usr/bin/env python3
import json
import re
from glob import glob
import argparse
import shutil
import os

def unescape_string(s):
    return s.replace('\\n', '\n').replace('\\t', '\t')

def migrate_file(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
        
    print(f"Processing: {filename}")
    
    # Create backup
    shutil.copyfile(filename, filename + ".bak")
    print(f"Created backup: {filename}.bak")
    
    # Load and migrate data
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for conv in data:
        for msg in conv['messages']:
            if 'content' in msg:
                msg['content'] = unescape_string(msg['content'])
            if 'logprobs' in msg:
                if 'content' in msg['logprobs']:
                    for token in msg['logprobs']['content']:
                        token['token'] = unescape_string(token['token'])
                        for top in token.get('top_logprobs', []):
                            if 'token' in top:
                                top['token'] = unescape_string(top['token'])
    
    # Save migrated file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Migrated: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Migrate prompt data to handle escaped newlines')
    parser.add_argument('files', nargs='+', help='JSON files to process')
    args = parser.parse_args()
    
    for pattern in args.files:
        for filename in glob(pattern):
            migrate_file(filename)
    print("Migration complete")

if __name__ == "__main__":
    main()
