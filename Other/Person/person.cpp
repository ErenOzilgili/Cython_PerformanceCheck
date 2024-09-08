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
    try{
        age = newAge;
        cout << "(C++) Age updated to " << newAge << endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "Caught an exception: " << e.what() << std::endl;
        throw std::runtime_error("An error occurred in C++");
    }
    
}

void Person::errorRaise(){
    throw std::runtime_error("Thrown from c++, person.cpp");
}