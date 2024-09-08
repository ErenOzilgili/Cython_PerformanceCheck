import wrappedPerson

#Address is a struct in c++, we will pass it as tuple.
address = ("Ulu", "Istanbul", "Marmara", 210)
name = "Eren"
age = 20

#Constructor
personObj = wrappedPerson.wPerson(name, age, address)

"""
Normal functionality shown
"""
#Update the age of the person
print("Updating age... with int = 21")
personObj.updateA(21)
personObj.display()

"""
Error thrown from cython and c++ respectively displayed above as examples
"""
print("\n------------------")
print("Error thrown from cython itself:")
print("Updating age... with string = \"a\"")
personObj.updateA("a")

print("\n--------------------")
print("Error thrown from c++ and caught in cython:")
personObj.exception()



