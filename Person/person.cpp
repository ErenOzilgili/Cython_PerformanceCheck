#include <iostream>
#include <string>
#include "person.h"

using namespace std;

// Constructor
Person::Person(const string& name, int age, const Address& address)
    : name(name), age(age), address(address) {}

// Method to display person details
void Person::displayPersonInfo(){
    cout << "Name: " << name << "\n"
                << "Age: " << age << "\n"
                << "Address: " << address.street << ", "
                << address.city << ", " << address.state << " "
                << address.zipCode << endl;
}

// Method to change the address
void Person::changeAddress(const Address& newAddress) {
    address = newAddress;
}

// Method to update age
void Person::updateAge(int newAge) {
    age = newAge;
    cout << "(C++) Age updated to " << newAge << endl;
}
