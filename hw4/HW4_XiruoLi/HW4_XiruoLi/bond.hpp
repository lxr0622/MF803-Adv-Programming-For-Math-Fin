#include <iostream>
using namespace std;

class Bond
{
private:
	double maturity; // maturity of bond
	double yield; // yield of bond
	double coupon; // coupon of bond
	double par; // par value of bond

public:
	//constructors
	Bond(); 
	Bond(double new_maturity, double new_yield, double new_coupon, double new_par);
	Bond(const Bond&b);
	~Bond();

	//bond pricing
	double price(double input_yield) const;

	//bond duration
	double duration(double h) const;

	//bond convexity
	double convexity(double h) const;

};
