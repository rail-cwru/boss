"""
Utility for config validation and refactoring.
"""

import os
import json
import glob
import argparse

from config import Config


def process_file(file):
    # Open file
    # Check for changes to file via cache or not in cache
    # If changes or not in cache
        # validate config
        # Modify symbol tracking table
        # Validate via class function
        # Update reference tree

    pass


def _load_symbol_table():
    # TODO cache symbol table file, key dict by filehash
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the config validation tool.')
    parser.add_argument('config_directory',
                        help='A path to the directory that contains config files. All subfolders will be seen.')
    parser.add_argument('--CLEAR_CACHE',
                        help='Clear cache.',
                        action='store_true')
    parser.add_argument('--CLEAR_SYMBOLS',
                        help='Clear symbol table.',
                        action='store_true')
    parser.add_argument('--CLEAR_REFERENCE',
                        help='Clear reference tree.',
                        action='store_true')
    args = parser.parse_args()

    path = args.config_directory

    # TODO If CLEAR_CACHE clear cache else load
    # If CLEAR_SYMBOL_TABLE clear symbol table else load
    symbol_table = _load_symbol_table()
    # If CLEAR_REFERENCE_TREE clear reftree else load

    # Load symbol tracking table

    # Walk files - for each file
        # Run process file

    # Now the cache is done
    # Now the symbol tracking table is done
    # Now the reference table is done

    # Open GUI

        # Display:
        # Reference graph view / Folder view
        # Highlight invalid entries

        # Actions:
        # Reorder reference graph
        # Reorder folder view
        # Rename symbols (ONLY IN JSONS)
        # Open config file in editor
        # Bulk editing
