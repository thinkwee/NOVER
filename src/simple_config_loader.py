#!/usr/bin/env python3

# Simple config loader for shell scripts.
# Usage:
# python simple_config_loader.py --config-name config --key key --default default

import os
import argparse
import yaml

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
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Navigate through nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        # Convert to string and return
        if value is not None:
            return str(value)
        else:
            return default
            
    except Exception as e:
        print(f"Error getting config value: {e}", file=os.sys.stderr)
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
    
    value = get_config_value(args.config_name, args.key, args.default or "")
    print(value)

if __name__ == "__main__":
    main() 