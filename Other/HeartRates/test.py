from wrap import *

firstName = input("Input your first name: ")

lastName = input("Input your last name: ")

day = int(input("Input the day: "))

month = day = int(input("Input the month: "))

year = day = int(input("Input the year: "))

wrappedHrObj = pyHR(firstName, lastName, day, month, year)

wrappedHrObj.setDa()

"""
cout << "\n****************************************************************" << endl;
	cout << "First Name: " << obj.getfirstName() << endl;
	cout << "Last Name: " << obj.getlastName() << endl;
	cout << "Date of Birth: " << obj.getbirthDay() << "/" << obj.getbirthMonth() << "/" << obj.getbirthYear() << endl;
	cout << "Age: " << obj.getAge() << endl;
	cout << "Maximum Hearth rate: " << obj.getMaximumHearthRate() << endl;
	cout << "Target Hearth Rate: " << obj.getMinimumTargetHearthRate() << "-" << obj.getMaximumTragetHearthRate() << endl;
"""

print("\n****************************************************************")
print("First Name: ", wrappedHrObj.getFName())
print("Last Name: ", wrappedHrObj.getLName())
print("Date of Birth: ", wrappedHrObj.getBDay())
print("Age: ", wrappedHrObj.getAgeOf())
print("Maximum Hearth rate: ", wrappedHrObj.getMaximumHR())
print("Target Hearth Rate: ", wrappedHrObj.getMinimumTargetHR())
