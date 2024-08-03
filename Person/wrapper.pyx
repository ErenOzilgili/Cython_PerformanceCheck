# distutils: language=c++
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
		void updateAge(int newAge)

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

	def updateA(self, int age):
		self.newP.updateAge(age)
		print("(Cython) You are {0} now".format(age))

	cdef Person.Address __arrangeAddrInfo(self, tuple address):
		cdef Person.Address addr
		addr.street = address[0].encode('utf-8')
		addr.city = address[1].encode('utf-8')
		addr.state = address[2].encode('utf-8')
		addr.zipCode = address[3]

		return addr
