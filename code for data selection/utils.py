import os
import getpass
import random
import numpy as np
from typing import Dict, Union, Type, List

def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        #print('the prefix---------',prefix)
        if os.path.exists(prefix):
            #print('the prefix====-----',prefix)
            return f"{prefix}/{getpass.getuser()}"
    
    os.makedirs(prefix)
    
    return f"{prefix}/{getpass.getuser()}"
    




