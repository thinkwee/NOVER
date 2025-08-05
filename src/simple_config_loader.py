#!/usr/bin/env python3

# Simple config loader for shell scripts.
# Usage:
# python simple_config_loader.py --config-name config --key key --default default

import os
import argparse
import subprocess

def get_config_value(config_name: str, key: str, default: str = "") -> str:
    """
    Get a value from a config file.

    Args:
        config_name: The name of the config file
        key: The key to get the value from
        default: The default value to return if the key is not found
        
    """
    config_file = f"config/{config_name}.yaml"
    
    if not os.path.exists(config_file):
        return default
    
    try:
        cmd = f"grep -E '^\\s*{key.split('.')[-1]}:\\s*' {config_file} | head -1 | awk -F': ' '{{print $2}}' | sed 's/#.*$//' | tr -d '\"' | xargs"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            return default
    except Exception as e:
        print(f"Error getting config value: {e}")
        return default

def main():
    """
    Simple config loader for shell scripts.

    Args:
        config_name: The name of the config file
        key: The key to get the value from
        default: The default value to return if the key is not found
    """
    parser = argparse.ArgumentParser(description="simple config loader for shell scripts")
    parser.add_argument("--config-name", default="config", help="config file name")
    parser.add_argument("--key", required=True, help="config key")
    parser.add_argument("--default", help="default value")
    
    args = parser.parse_args()
    
    key_parts = args.key.split('.')
    last_key = key_parts[-1]
    
    value = get_config_value(args.config_name, last_key, args.default or "")
    print(value)

if __name__ == "__main__":
    main() 