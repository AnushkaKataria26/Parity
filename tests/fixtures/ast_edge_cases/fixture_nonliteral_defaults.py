import logging
import os

TIMEOUT = 30

def f1(x=TIMEOUT):                    # name reference
    pass

def f2(x=logging.INFO):               # attribute access
    pass

def f3(x=os.environ.get("X")):        # call expression
    pass

def f4(x=[1, 2, 3]):                  # literal list — ast.literal_eval CAN handle this
    pass

def f5(x=(1, 2)):                     # literal tuple — literal_eval CAN handle this
    pass

def f6(x={1: 2}):                     # literal dict — literal_eval CAN handle this
    pass

def f7(x=-5):                         # unary minus on a literal — literal_eval CAN handle this
    pass

def f8(x=1 + 2):                      # binary expression — literal_eval CANNOT handle this
    pass
