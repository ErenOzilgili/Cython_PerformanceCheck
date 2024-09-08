#ifndef PERSON_H
#define PERSON_H

//Header for the person class

#include <iostream>
#include <string>

using namespace std;

class Person{
public:
	struct Address{
        string street;
        string city;
        string state;
        int zipCode;

        Address() = default;  // Default constructor
    };

	string name;
	Address address;
	int age;

	Person(const string& name, int age, const Address& address);

	void displayPersonInfo();
	void changeAddress(const Address& newAddress);
	void updateAge(int newAge);
    void errorRaise();
};

#endif