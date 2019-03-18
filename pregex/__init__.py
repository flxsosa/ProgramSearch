import sys
if sys.version_info.major == 3:
    from .pregex import *
else:
    from pregex import *
