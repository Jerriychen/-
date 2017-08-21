class student(object):
	def __init__(self,name,age,score):
		self.__name = name
		self.__age = age
		self.__score = score
	@property
	def score(self):
		pass
	@score.setter
	def score(self,score):
		if ~isinstance(score,int):
			raise ValueError('score must be an integer!')
		if score>100 or score<0:
			raise ValueError('score must betweet 0 to 100')
		self.__score = score
	@score.getter
	def score(self):
		return self.__score

	@property
	def name(self):
		return self.__name
	@name.setter
	def name(self,name):
		self.__name = name
	@name.getter
	def name(self):
		return self.__name
	@name.deleter
	def name(self):
		pass

import functools
def debug(func):
	@functools.wraps(func)
	def wrapper(*args,**kwargs):
		print 'execute %s hahahahhaha'%(func.__name__)
		return func(*args,**kwargs)
	return wrapper

@debug
def say_hello(name='chen',age=23,score=80,**kwargs):
	print 'hello'
	print name,age,score
	print kwargs
# say_hello('chenzezhong',22,98,city = 'jinan')
say_hello(city = 'jinan')
print say_hello.__name__