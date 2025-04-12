#!/usr/bin/env python
"""
顔認識自動クロップツールのスタートアップスクリプト
"""

import os
import sys

# srcディレクトリをPythonパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from auto_crop_gui import main

if __name__ == '__main__':
    main()
