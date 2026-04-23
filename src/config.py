# src/config.py
import fastf1

CACHE_DIR = "data/raw/fastf1_cache"
fastf1.Cache.enable_cache(CACHE_DIR)

# Target session
YEAR   = 2024
ROUND  = 1          # Bahrain GP
SESSION = "R"       # Race