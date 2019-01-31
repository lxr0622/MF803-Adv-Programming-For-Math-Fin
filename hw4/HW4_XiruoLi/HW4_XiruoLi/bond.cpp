#include "bond.hpp"
#include <cmath>
using namespace std;

//default constructor
Bond::Bond()
{
	maturity = 1;
	yield = 0.025;
	coupon = 0;
	par = 100;
}

//parameter consturctor
Bond::Bond(double new_maturity, double new_yield, double new_coupon, double new_par)
{
	maturity = new_maturity;
	yield = new_yield;
	coupon = new_coupon;
	par = new_par;
}

//copy constructor
Bond::Bond(const Bond&b)
{
	maturity = b.maturity;
	yield = b.yield;
	coupon = b.coupon;
	par = b.par;
}

//destructor
Bond::~Bond()
{

}

//bond pricing
double Bond::price(double m_yield) const
{
	double price = 0.0;
	for (int i = 1; i <= maturity; ++i) {
		price += coupon / pow(1 + m_yield, i);
	}
	price += par / pow(1 + m_yield, maturity);
	return price;
}

//bond duration (Approximated Macauly Duration)
double Bond::duration(double h) const
{
	return (price(yield - h) - price(yield + h)) / (2 * price(yield) * h ) * (1 + yield);
}

//bond convexity
double Bond::convexity(double h) const
{
	return (price(yield +h) + price(yield -h) - 2 * price(yield)) / (h * h * price(yield));
}
