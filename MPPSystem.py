#!/usr/bin/python

import os
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import pickle
from datetime import *
import csv
from pylab import *
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

#the class is an MPP System for the dev specified at initialization
Base = declarative_base()

class States(Base):
	__tablename__ = 'states'
	id_ = Column(Integer, autoincrement=True, primary_key=True)
	time = Column(Integer, nullable=False)
	level = Column(Integer,nullable=False)
	drop = Column(Integer, nullable=False)
	fg = Column(Float, nullable=False)

	observed = relationship("ObservedData")
	prob	 = relationship("TransitionMatrix") 

	def __init__(self, time, level, drop, fg):
		self.time = time
		self.level = level
		self.drop = drop
		self.fg = fg

	def __repr__(self):
		return "<State(time="+self.time+", level="+self.level+", drop="+self.drop+", fg="+self.fg+")>" 

class ObservedData(Base):
	__tablename__ = 'observed_data'
	id_ = Column(Integer, autoincrement=True, primary_key=True)
	state_id = Column(Integer,ForeignKey('states.id_'))
##state_id= Column(Integer, nullable=False)
	t_left = Column(Integer, nullable=False)

	#states = relationship("States", back_populates="observed")
	
	def __init__(self, state_id, t_left):
		self.state_id = state_id
		self.t_left = t_left

	def __repr__(self):
		return "<ObservedData(state id="+self.state_id+", time left="+self.t_left+")>" 

class TransitionMatrix(Base):
	__tablename__ = 'transition'
	id_ = Column(Integer, autoincrement=True, primary_key=True)
	state_id_now = Column(Integer,ForeignKey('states.id_'))
	state_id_next = Column(Integer,ForeignKey('states.id_'))
	seen = Column(Integer, default=0, nullable=False)
	total = Column(Integer, default,nullable=False)

	def __init__(self,state_id_now, state_id_next):
		self.state_id_now = state_id_now
		self.state_id_next = state_id_next

	def __repr__(self):
		 return "<TransitionMatrix(current state id="+self.state_id_now+", next state id="+self.state_id_next+", # of occurences="+self.seen+", total # of occurrences=", self.total+")>"

class MPPSystem:
	Session = sessionmaker()
	path_to = '/home/anudipa/Documents/Jouler_Extra/scripts/data/shortlisted'
	path_to_csv = '/home/anudipa/Documents/Jouler_Extra/scripts/data/csv'
	def __init__(self, dev):
		#connect to database
		self.engine = create_engine('mysql+mysqldb://anudipa:db8287@localhost/mpp', echo=True)
		connection = self.engine.connect()
		Base.metadata.create_all(self.engine)
		self.Session = sessionmaker()
		self.Session.configure(bind=self.engine)
		self.dev = dev

	def create_state(self):
		session = self.Session()
		data_ = []
		for t in range(30,1801,30):
			for l in range(0,101,5):
				for d in range(0,101,5):
					data_.append(States(time=t, level=l, drop=d, fg=0.01))
					f = 0.0
					while(f <= 1):
						data_.append(States(time=t, level=l, drop=d, fg=f))
						f += 0.1
		session.add_all(data_)
		session.commit()

		#check what is inserted
		#for row in session.query(States).filter_by(time=30):
		#	print(row.time, row.level, row.drop, row.fg)
		
		session.close()

	def query_state(self):
		session = self.Session()
		for row in session.query(States).filter_by(time==30):
			print(row.id_, row.time, row.level, row.drop, row.fg)
		session.close()

	def query_obs(self):
		session = self.Session()
		for row in session.query(ObservedData).filter_by(t_left > 1500):
			print(row.states_id, row.t_left)

	def fill_data(self):
		session = self.Session()
		#pdata = pickle.load(open(os.join(path_to, dev+'.p'), 'rb'), encoding='bytes')
		csvF = os.path.join(self.path_to_csv, self.dev+'.csv')
		data_ = []
		i = 0
		with open(csvF) as csvfile:
			reader = csv.reader(csvfile, delimiter=';')
			heading = next(reader) 
			with self.engine.connect() as con:
				for row in reader:
					#print(row)
					t = int(row[0])
					l = int(row[1])
					d = int(row[2])
					f = float(row[3])
					tl = int(row[4])
					if t < 30 or t > 1800:
						print('!!!!!!!!!!!!', row)
						continue
					if f == 0.01:
						stmt = ("select id_ from states where time=%d and level=%d and states.drop=%d and states.fg like %.2f;"%(t,l,d,f))
					elif f==0 or f==1:
						stmt = ("select id_ from states where time=%d and level=%d and states.drop=%d and states.fg=%d;"%(t,l,d,f))
					else:
						stmt = ("select id_ from states where time=%d and level=%d and states.drop=%d and states.fg like %.1f;"%(t,l,d,f))
					rs = con.execute(stmt)
					if rs == None:
						continue
					id_ = rs.first()[0]
					#print('#',i, rs.first()[0], t, l, d, f, tl)
					i += 1
					data_.append(ObservedData(state_id=id_, t_left=tl))
					if i > 1500:
						break
				con.close()
			session.add_all(data_)
			session.commit()

#fill all possible state transitions in init, then fill in occurrences for each transition
	def init_transition_matrix(self):
		session = self.Session()
		
#select for each stateid<i>, get all stateid<j> where t<i> <= t<j> , l<i> >= l<j>
		

	def exit_session(self, session):
		session.close()
