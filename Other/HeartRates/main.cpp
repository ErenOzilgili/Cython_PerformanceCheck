#include <iostream>
#include "HeartRates.h"
using namespace std;

int main() {
	string firstName;
	string lastName;
	int day;
	int month;
	int year;

	cout << "Please enter your first and last name here: ";
	cin >> firstName >> lastName;

	cout << "Please enter day, month, year of your birth: ";
	cin >> day >> month >> year;

	//Create HearthRates object
	HeartRates obj(firstName, lastName, day, month, year);
	obj.setDate();
	cout << "\n****************************************************************" << endl;
	cout << "First Name: " << obj.getfirstName() << endl;
	cout << "Last Name: " << obj.getlastName() << endl;
	cout << "Date of Birth: " << obj.getbirthDay() << "/" << obj.getbirthMonth() << "/" << obj.getbirthYear() << endl;
	cout << "Age: " << obj.getAge() << endl;
	cout << "Maximum Hearth rate: " << obj.getMaximumHearthRate() << endl;
	cout << "Target Hearth Rate: " << obj.getMinimumTargetHearthRate() << "-" << obj.getMaximumTragetHearthRate() << endl;

	return 0;
}