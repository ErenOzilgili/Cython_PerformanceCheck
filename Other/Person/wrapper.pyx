from libcpp.string cimport string

cdef extern from "person.h":
	cdef cppclass Person:
		struct Address:
			string street
			string city
			string state
			int zipCode

			Address()

		Person(const string& name, int age, const Address& address)

		string name
		Address address
		int age

		void displayPersonInfo()
		void changeAddress(const Address& newAddress)
		void updateAge(int newAge) except +
		void errorRaise() except +

cdef class wPerson:
	cdef Person* newP

	#Constructor
	def __cinit__(self, name, age, address):
		#Decompose address
		cdef Person.Address addr = self.__arrangeAddrInfo(address)
		self.newP = new Person(name.encode('utf-8'), age, addr)

	#Destructor
	def __dealloc__(self):
		del self.newP

	def display(self):
		self.newP.displayPersonInfo()

	def changeAddr(self, address):
		cdef Person.Address addr = self.__arrangeAddrInfo(address)
		self.newP.changeAddress(addr)

	def updateA(self, age):
		try:
			self.newP.updateAge(age)
			print("(Cython) You are {0} now".format(age))
		except TypeError as e:
			print("Error has been encountered! String is passed when expected int")
			#raise RuntimeError(e)
		except RuntimeError as e:
			print("Error has been encountered! Runtime error --- division by zero here in this example")
			#raise RuntimeError(e)
		else:
			print("Couldn't catch the error type")

	def exception(self):
		try:
			self.newP.errorRaise()
		except RuntimeError as e:
			print("RuntimeError --- Printed in try-except block except part")
			#raise RuntimeError(e)
		else:
			print("Did catch but not runtime_error!")

	def exception2(self):
		try:
			self.newP.errorRaise()
		finally:
			pass

	cdef Person.Address __arrangeAddrInfo(self, tuple address):
		cdef Person.Address addr
		addr.street = address[0].encode('utf-8')
		addr.city = address[1].encode('utf-8')
		addr.state = address[2].encode('utf-8')
		addr.zipCode = address[3]

		return addr
