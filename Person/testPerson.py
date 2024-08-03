import wrappedPerson

#Address is a struct in c++, we will pass it as tuple.
address = ("Ulu", "Istanbul", "Marmara", 210)
name = "Eren"
age = 20

#Constructor
personObj = wrappedPerson.wPerson(name, age, address)

#Update the age of the person
personObj.updateA(21)