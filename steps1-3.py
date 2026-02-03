# %% 

# **Step 1**

# **College Dataset**

# Problems that could be addressed using this dataset: Which characteristics improve student success, comparing outcomes between different institutions, predicting which institutions have better completion rates
# Main question: What characteristics predict whether a college has a high graduation rate?

# **Job Placement Dataset**

# Problems that could be addressed using this dataset: Predicting employment outcomes, identifying at risk individuals, understanding which academic stage matters most for employability
# Main question: Which factors contribute most to job placement?
# %%
# **Step 2**

# **College Dataset**

# Main question: What characteristics predict whether a college has a high graduation rate?
# IBM: Graduation rate

# **Job Placement Dataset**
# Main question: Which factors contribute most to job placement?
# IBM: Placement rate

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

college_df = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv")
placement_df = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")



# %%
# **Step 3**