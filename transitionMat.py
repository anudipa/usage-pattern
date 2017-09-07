#!/usr/bin/python

import os
import numpy as np
from collections import defaultdict,Counter, OrderedDict
import pickle
from datetime import *
import csv
from pylab import *
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

#tables 

#States, transition
