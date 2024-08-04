from libcpp.string cimport string
cimport cython

cdef extern from "HeartRates.h":
	cdef cppclass HeartRates:
		HeartRates(string fN, string lN, int d, int m, int y)
		void setfirstName(string fN)
		string getfirstName()
		void setlastName(string lN)
		string getlastName()
		void setbirhtDay(int d)
		int getbirthDay()
		void setbirthMonth(int m)
		int getbirthMonth()
		void setbirthYear(int y)
		int getbirthYear()
		void setDate()
		int getAge()
		int getMaximumHearthRate()
		int getMinimumTargetHearthRate()
		int getMaximumTragetHearthRate()

cdef class pyHR:
	cdef HeartRates* hrObj
	
	def __cinit__(self, fN, lN, int d, int m, int y):
		cdef string firstN = fN.encode('utf-8')
		cdef string lastN = lN.encode('utf-8')
		self.hrObj = new HeartRates(firstN, lastN, d, m ,y)

	def __dealloc__(self):
		del self.hrObj

	def setFName(self, fN):
		cdef string firstN = fN.encode('utf-8')
		self.hrObj.setfirstName(fN)

	def getFName(self):
		return self.hrObj.getfirstName().decode('utf-8')

	def setLName(self, lN):
		cdef string lastN = lN.encode('utf-8')
		self.hrObj.setlastName(lastN)

	def getLName(self):
		return self.hrObj.getlastName().decode('utf-8')

	def setBDay(self, int d):
		self.hrObj.setbirhtDay(d)

	def getBDay(self):
		return self.hrObj.getbirthDay()

	def setBMonth(self, int m):
		self.hrObj.setbirthMonth(m)

	def getBDay(self):
		return self.hrObj.getbirthMonth()

	def setBYear(self, int y):
		self.hrObj.setbirthYear(y)

	def getBYear(self):
		return self.hrObj.getbirthYear()

	def setDa(self):
		self.hrObj.setDate()

	def getAgeOf(self):
		return self.hrObj.getAge()	 

	def getMaximumHR(self):
		return self.hrObj.getMaximumHearthRate()

	def getMaximumTargetHR(self):
		return self.hrObj.getMaximumTragetHearthRate()

	def getMinimumTargetHR(self):
		return self.hrObj.getMinimumTargetHearthRate()