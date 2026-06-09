"""Put the plugin root on sys.path so ``import resolver`` works in tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
