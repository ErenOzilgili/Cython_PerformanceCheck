#ifndef HEARTRATES_H
#define HEARTRATES_H

using namespace std;
#include <string>

class HeartRates {
public:
	HeartRates(string fN, string lN, int d, int m, int y);
	void setfirstName(string fN);
	string getfirstName();
	void setlastName(string lN);
	string getlastName();
	void setbirhtDay(int d);
	int getbirthDay();
	void setbirthMonth(int m);
	int getbirthMonth();
	void setbirthYear(int y);
	int getbirthYear();
	void setDate();
	int getAge();
	int getMaximumHearthRate();
	int getMinimumTargetHearthRate();
	int getMaximumTragetHearthRate();
private:
	std::string firstName;
	std::string lastName;
	int birthDay{ 0 };
	int birthMonth{ 0 };
	int birthYear{ 0 };
	int currentDay{ 0 };
	int currentMonth{ 0 };
	int currentYear{ 0 };
};

#endif