#include "bond.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
using namespace std;

int main() 
{
	//(a) price of zero coupon bond
	cout << "problem (a)" << endl;
	vector<int> maturity_vec{ 1,2,3,5,10,30 }; // maturity
	vector<double> yield_vec{ 0.025,0.026,0.027,0.03,0.035,0.04 }; // yield 
	vector<string> name_vec{ "1 year maturity bond", "2 year maturity bond","3 year maturity bond","5 year maturity bond","10 year maturity bond","30 year maturity bond" };
	vector<Bond> bond_vec1; // zero coupon bond
	for (int i = 0; i < 6; ++i) {
		bond_vec1.push_back(Bond(maturity_vec[i], yield_vec[i], 0, 100));
		cout << "Price of " << name_vec[i] << ": " << bond_vec1[i].price(yield_vec[i]) << endl;
	}

	//(b) duration of zero coupon bond
	cout << "problem (b)" << endl;
	double dy = 0.001;
	for (int i = 0; i < 6; ++i) {
		cout << "Apprximated Macauly Duration of " << name_vec[i] << ": " << bond_vec1[i].duration(dy) << endl;
	}

	//(c) price of coupon bond
	cout << "problem (c)" << endl;
	vector<Bond> bond_vec2; // coupon bond
	for (int i = 0; i < 6; ++i) {
		bond_vec2.push_back(Bond(maturity_vec[i], yield_vec[i], 3, 100));
		cout << "Price of " << name_vec[i] << " with 3% coupon" << ": " << bond_vec2[i].price(yield_vec[i]) << endl;
	}

	//(d) duration of coupon bond
	cout << "problem (d)" << endl;
	for (int i = 0; i < 6; ++i) {
		cout << "Apprximated Macauly Duration of " << name_vec[i] << " with 3% coupon" << ": " << bond_vec2[i].duration(dy) << endl;
	}

	//(e) convexity of bond
	cout << "problem (e)" << endl;
	for (int i = 0; i < 6; ++i) {
		cout << "Apprximated Convexity of " << name_vec[i] << ": " << bond_vec1[i].convexity(dy) << endl;
		cout << "Apprximated Convexity of " << name_vec[i] << " with 3% coupon" << ": " << bond_vec2[i].convexity(dy) << endl;
	}

	//(f) initial value of portfolio
	cout << "problem (f)" << endl;
	double port_price0 = bond_vec1[0].price(yield_vec[0]) + bond_vec1[2].price(yield_vec[2]) - 2 * bond_vec1[1].price(yield_vec[1]);
	cout << "Initial value of portfolio is: " << port_price0 << endl;

	//(g) duration and convexity of portfolio
	cout << "problem (g)" << endl;
	double port_price1 = bond_vec1[0].price(yield_vec[0] - dy) + bond_vec1[2].price(yield_vec[2] - dy) - 2 * bond_vec1[1].price(yield_vec[1] - dy);
	double port_price2 = bond_vec1[0].price(yield_vec[0] + dy) + bond_vec1[2].price(yield_vec[2] + dy) - 2 * bond_vec1[1].price(yield_vec[1] + dy);
	double port_duration = (port_price1 - port_price2) / (2 * port_price0 * dy) * (1 + yield_vec[1]);
	double port_convexity = (port_price1 + port_price2 - 2 * port_price0) / (pow(dy, 2)  * port_price0);
	cout << "Duration of portfolio is: " << port_duration << endl;
	cout << "Convexity of portfolio is: " << port_convexity << endl;

	//(h) yield rises by 0.01
	cout << "problem (h)" << endl;
	double dy1 = 0.01;
	double port_price3 = bond_vec1[0].price(yield_vec[0] + dy1) + bond_vec1[2].price(yield_vec[2] + dy1) - 2 * bond_vec1[1].price(yield_vec[1] + dy1);
	cout << "When yield increase by 1%, value of portfolio is: " << port_price3 << endl;

	//(i) yield decreases by 0.01
	cout << "problem (i)" << endl;
	double port_price4 = bond_vec1[0].price(yield_vec[0] - dy1) + bond_vec1[2].price(yield_vec[2] - dy1) - 2 * bond_vec1[1].price(yield_vec[1] - dy1);
	cout << "When yield decrease by 1%, value of portfolio is: " << port_price4 << endl;

	//(j) amortizing bond
	cout << "problem (j)" << endl;
	vector<double> cashflow;
	for (int i = 1; i <= 5; ++i) {
		cout << "Cashflow of year " << i << ": " << 23/ pow(1 + 0.03, i) << endl;
		cashflow.push_back(23 / pow(1 + 0.03, i));
	}

	//(k) price and duration of amortizing bond
	cout << "problem (k)" << endl;
	Bond amor_bond(5, 0.03, 23, 0);
	cout << "Price of amortizing bond is: " << amor_bond.price(yield_vec[3]) << endl;
	cout << "Duration of amortizing bond is: " << amor_bond.duration(dy) << endl;

	system("pause");
	return 0;
}