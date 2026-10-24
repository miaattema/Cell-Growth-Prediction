# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================
import os
import pandas as pd

from functions import growth_range

PH_LIMITS = (6.8, 7.2)
TEMPERATURE_LIMITS = (35.0, 37.0)
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

