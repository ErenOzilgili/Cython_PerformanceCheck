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
Both has been catched inside cython
"""
print("\n------------------")
print("Error thrown from cython itself:")
print("Updating age... with string = \"a\"")
personObj.updateA("a")

print("\n--------------------")
print("Error thrown from c++ and caught in cython:")
personObj.exception()

"""
Error thrown from c++ and caught in python in example below
"""
print("\n--------------------")
print("Error thrown from c++ and caught in python:")
try:
    personObj.exception2()
except RuntimeError as e:
    print("Catched exception thrown from c++ in python")
else:
    print("Did not catch runtime error!")


