import wrappedPerson

#Address is a struct in c++, we will pass it as tuple.
address = ("Ulu", "Istanbul", "Marmara", 210)
name = "Eren"
age = 20

#Constructor
personObj = wrappedPerson.wPerson(name, age, address)

#Update the age of the person
personObj.updateA(21)

#Print the address information of a given person
wrappedPerson.LocationTracker(personObj)

#Change the name information of a given person and return the previous name as a str object
old_name = wrappedPerson.NameChange(personObj,"Efe")
print(old_name)
