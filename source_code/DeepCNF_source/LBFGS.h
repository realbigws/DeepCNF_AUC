#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;
double LBFGS_rt = 1e-8;

class Minimizer 
{
	bool use_preconditioner;
	int N;
	int M;
	double term_ratio;
	double alpha;
	double beta;
	double gamma1;
	double gamma2;
	double min_improvement_ratio;
	int max_line_search_evaluations;

	virtual void Report(const vector<double> &theta, int iteration,
			double objective, double step_length) = 0;
	virtual void Report(const string &s) = 0;
	virtual void ComputeGradient(vector<double> &g, const vector<double> &x) = 0;
	virtual void ComputeHessianDiagonal(vector<double> &h,const vector<double> &x) {h = x;}
	virtual double ComputeFunction(const vector<double> &x) = 0;

	double LineSearch(const vector<double> &x, const vector<double> &d,
			const vector<double> &g, double &f_curr);
public:

	Minimizer(bool use_preconditioner=0, const int M = 20, /* number of previous gradients to remember                          */
	const double term_ratio = 0.000001, /* required ratio of gradient norm to parameter norm for termination */
	const double alpha = 0.000001, /* minimum improvement ratio for sufficient decrease                 */
	const double beta = 0.5, /* default step size                                                 */
	const double gamma1 = 0.01, /* maximum step size                                                 */
	const double gamma2 = 0.8, /* minimum step size                                                 */
	const double min_improvement_ratio = 1.000001, /* minimum improvement ratio after sufficient decrease               */
	const int max_line_search_evaluations = 10); /* maximum number of line search function evaluations                */

	virtual ~Minimizer() {
	}

	void LBFGS(vector<double> &x0, /* initial guess of solution    */
	const int max_iterations = 100); /* maximum number of iterations */

	void ApproximateGradient(vector<double> &g, const vector<double> &x,
			int start,int end,const double EPSILON = 1e-4);
};

/* Standard linear algebra */
double DotProduct(const vector<double> &x, const vector<double> &y);
double Norm(const vector<double> &x);
const vector<double>
		operator/(const vector<double> &x, const vector<double> &y);
const vector<double>
		operator+(const vector<double> &x, const vector<double> &y);
const vector<double>
		operator-(const vector<double> &x, const vector<double> &y);
const vector<double> &operator+=(vector<double> &x, double c);
const vector<double> operator*(const vector<double> &x, double c);

/* Constructor */
Minimizer::Minimizer(bool use_preconditioner, const int M,
		const double term_ratio, const double alpha, const double beta,
		const double gamma1, const double gamma2,
		const double min_improvement_ratio,
		const int max_line_search_evaluations) :
	use_preconditioner(use_preconditioner=0), M(M), term_ratio(term_ratio),
			alpha(alpha), beta(beta), gamma1(gamma1), gamma2(gamma2),
			min_improvement_ratio(min_improvement_ratio),
			max_line_search_evaluations(max_line_search_evaluations) {
}

/* Modified cubic backtracking line search */
double Minimizer::LineSearch(const vector<double> &x, const vector<double> &d,
		const vector<double> &g, double &f_curr) 
{
	int num_evaluations = 0;
	const double dot_prod = DotProduct(d, g);
	bool increasing_step = false;
	bool sufficient_decrease = false;
	double best_t = 0;
	double best_f = f_curr;

	/* First, try a full Newton step. */
	double t_new1 = 1;
	double f_new1 = ComputeFunction(x + d * t_new1);
	if (f_new1 < best_f)
	{
		best_f = f_new1;
		best_t = t_new1;
	}
	if (f_new1 <= f_curr + alpha * t_new1 * dot_prod)
		sufficient_decrease = true;

	/* If a sufficient decrease is found, then we'll allow the multiplier to get larger.
	 * Otherwise, we'll force the multiplier to get gradually smaller. */
	if (sufficient_decrease)
		increasing_step = true;

	/* Now perform quadratic interpolation of:
	 *
	 *    f_curr + dot_prod * t + ((f_new1 - f_curr) / t_new1^2 - dot_prod / t_new1) * t^2
	 *
	 * Note that this function is equal to f_curr at t = 0 and f_new1 at t = t_new1.  Note
	 * also that the quadratic fit works only if the coefficient of t^2 is positive; this
	 * will always be the case when a sufficient decrease has not been found since
	 *
	 *    f_new1 > f_curr + alpha * t_new1 * dot_prod
	 *
	 * implies
	 *
	 *    f_new1 > f_curr + t_new1 * dot_prod
	 */

	double t_new2 = t_new1;
	double f_new2 = f_new1;
	t_new1 = -dot_prod / (2 * ((f_new2 - f_curr) / t_new2 - dot_prod) / t_new2);

	/* Check to make sure the minimization of the quadratic was valid.  If not, try scaling the
	 * t value instead. */
	if (f_new2 <= f_curr + dot_prod * t_new2) 
	{
		if (increasing_step)
			t_new1 = t_new2 / beta;
		else
			t_new1 = t_new2 * beta; /* This case is not really necessary, as explained above. */
	}

	/* If we're doing a decreasing step, clip the prediction to a restricted range. */
	if (!increasing_step)
		t_new1 = max(gamma1 * t_new2, min(gamma2 * t_new2, t_new1));

	/* Compute the new function value, check for sufficient decrease, and check for termination. */
	f_new1 = ComputeFunction(x + d * t_new1);
	if (f_new1 < best_f) 
	{
		best_f = f_new1;
		best_t = t_new1;
	}
	if (f_new1 <= f_curr + alpha * t_new1 * dot_prod)
	{
		sufficient_decrease = true;
	}
	if (sufficient_decrease && f_new1 >= f_new2 * min_improvement_ratio) 
	{
		f_curr = best_f;
		return best_t;
	}

	//--- while loop ---//
	while (true) 
	{
		/* Now perform cubic interpolation of
		 *
		 *    f_curr + dot_prod * t + b * t^2 + a * t^3
		 */
		double a = 1 / (t_new1 - t_new2) * ((f_new1 - f_curr - dot_prod
				* t_new1) / (t_new1 * t_new1) - (f_new2 - f_curr - dot_prod
				* t_new2) / (t_new2 * t_new2));
		double b = 1 / (t_new1 - t_new2) * (-(f_new1 - f_curr - dot_prod
				* t_new1) * t_new2 / (t_new1 * t_new1) + (f_new2 - f_curr
				- dot_prod * t_new2) * t_new1 / (t_new2 * t_new2));
		t_new2 = t_new1;
		f_new2 = f_new1;
		t_new1 = (-b + sqrt(b * b - 3 * a * dot_prod)) / (3 * a);

		/* Check to make sure the minimization of the cubic was valid.  If not, try scaling the
		 * t value instead. */
		if (b * b - 3 * a * dot_prod <= 0) 
		{
			if (increasing_step)
				t_new1 = t_new2 / beta;
			else
				t_new1 = t_new2 * beta;
		}

		/* If we're doing a decreasing step, clip the prediction to a restricted range. */
		if (!increasing_step)
			t_new1 = max(gamma1 * t_new2, min(gamma2 * t_new2, t_new1));

		/* Compute the new function value, check for sufficient decrease, and check for termination. */
		f_new1 = ComputeFunction(x + d * t_new1);
		if (f_new1 < best_f) 
		{
			best_f = f_new1;
			best_t = t_new1;
		}
		if (f_new1 <= f_curr + alpha * t_new1 * dot_prod)
			sufficient_decrease = true;
		if (sufficient_decrease && f_new1 * min_improvement_ratio >= f_new2) 
		{
			f_curr = best_f;
			return best_t;
		}
		if (++num_evaluations >= max_line_search_evaluations) 
		{
			f_curr = best_f;
			return best_t;
		}
	}
}

/* Finite-difference--based gradient */
void Minimizer::ApproximateGradient(vector<double> &g, const vector<double> &x, 
	int start,int end, const double EPSILON) 
{
	double base = ComputeFunction(x);
	vector<double> x_copy = x;
	for (int i = start; i <= end; i++) 
	{
		x_copy[i] += EPSILON;
		g[i] = (ComputeFunction(x_copy) - base) / EPSILON;
		x_copy[i] = x[i];
	}
}

/* LBFGS routine */
void Minimizer::LBFGS(vector<double> &x0, const int max_iterations)
{
	/* Initialization */
	int N = x0.size();
	vector<vector<double> > x(2, vector<double> (N, 0.0)); /* iterates                    */
	vector<vector<double> > g(M, vector<double> (N, 0.0)); /* gradients                   */
	vector<vector<double> > y(M, vector<double> (N, 0.0)); /* y[k] = g[k+1] - g[k]        */
	vector<vector<double> > s(M, vector<double> (N, 0.0)); /* s[k] = x[k+1] - x[k]        */
	vector<double> d(N, 0.0); /* d[k] = -H[k] g[k]           */
	vector<double> rho(M, 0.0); /* rho[k] = 1 / (y[k]^T s[k])  */
	vector<double> a(M, 0.0);
	vector<double> b(M, 0.0);
	vector<double> h(N, 0.0); /* hessian diagonal            */

	double f_prev = 0;
	double f_curr = ComputeFunction(x0);
	int k = 0, iterations = 0;
	x[0] = x0;
	int num_consec_small_steps = 0;
	bool progress_made = true;

	//-- init check --//
	if(x0.size() == 0)
	{
		fprintf(stderr,"Empty initial vector specified.");
		exit(-1);
	}

	//-- while loop --//
	while (true) 
	{
		/* STEP ONE: Compute new gradient vector */
		ComputeGradient(g[k % M], x[k % 2]);
		if (use_preconditioner)
			ComputeHessianDiagonal(h, x[k % 2]);

		/* STEP TWO: Check termination conditions */
		/*
		 if (Norm (g[k%M]) < term_ratio * max (1.0, Norm (x[k%2]))){
		 oss.str("");
		 oss << "Termination condition: gradient vector small ("
		 << Norm (g[k%M]) / max (1.0, Norm (x[k%2])) << " < " << term_ratio << ")";
		 Report (oss.str());
		 break;
		 }*/

		/* STEP THREE: Update iterates */
		if (k > 0)
			y[(k - 1 + M) % M] = g[k % M] - g[(k - 1 + M) % M];
		if (k > 0)
			s[(k - 1 + M) % M] = x[k % 2] - x[(k - 1 + 2) % 2];
		if (k > 0)
			rho[(k - 1 + M) % M] = 1.0 / DotProduct(y[(k - 1 + M) % M], s[(k - 1 + M) % M]);

		/* STEP FOUR: Compute new search direction */
		d = g[k % M];
		if ((k > 0 && DotProduct(y[(k - 1 + M) % M], s[(k - 1 + M) % M]) <= 0) || k > 200) 
		{
			/* Delete the old gradient info */
			g[0] = g[k % M];
			x[0] = x[k % 2];
			k = 0;
		}
		//----- calculate d ------//
		{
			/* Use Nocedal's recursion to compute H_k g_k */
			for (int j = k - 1; j >= max(0, k - M); j--) 
			{
				a[j % M] = DotProduct(s[j % M], d) * rho[j % M];
				d = d - y[j % M] * a[j % M];
			}

			/* Apply preconditioner (inverse Hessian diagonal) */
			if (use_preconditioner) 
				d = d / h;
			else 
				d = d * (1.0 / Norm(g[k % M]));

			/* Continue using recursion formula */
			for (int j = max(0, k - M); j <= k - 1; j++) 
			{
				b[j % M] = DotProduct(y[j % M], d) * rho[j % M];
				d = d + s[j % M] * (a[j % M] - b[j % M]);
			}
		}
		d = d * -1.0;

		/* STEP FIVE: Do line search, update f_curr, and take step */
		f_prev = f_curr;
		double step = LineSearch(x[k % 2], d, g[k % M], f_curr);
		x[(k + 1) % 2] = x[k % 2] + d * step;

		iterations++;
		k++;

		/* STEP SIX: Check termination conditions */
		if(iterations%1==0) Report(x[k % 2], iterations, f_curr, step);
		if (iterations >= max_iterations) 
		{
			cerr << "Termination: maximum number of iterations reached" << endl;
			break;
		}

		if (f_curr == 0) 
		{
			cerr << "Termination: Zero reached." << endl;
			break;
		}

		if (fabs(f_prev - f_curr) / f_prev < LBFGS_rt)
		{
			num_consec_small_steps++;
		}
		else 
		{
			num_consec_small_steps = 0;
			progress_made = true;
		}
		progress_made = true;

		//--- change LBFGS_rt or not ----//
		if (num_consec_small_steps == 10) 
		{
			if (progress_made) 
			{
				progress_made = false;
				num_consec_small_steps = 0;
				cerr << "Restart: Too many consecutive small steps" << endl;
				g[0] = g[k % M];
				x[0] = x[k % 2];
				LBFGS_rt = max(LBFGS_rt / 2, 1e-8);
				k = 0;
			} 
			else 
			{
				cerr << "Termination: Too many consecutive small steps" << endl;
				break;
			}
		}
	}
	x0 = x[k % 2];
}

/* Standard linear algebra */
double DotProduct(const vector<double> &x, const vector<double> &y) 
{
	if(x.size() != y.size())
	{
		fprintf(stderr,"Vector size mismatch.");
		exit(-1);
	}
	double ret = 0;
	for (int i = 0; i < (int) x.size(); i++) ret += x[i] * y[i];
	return ret;
}
double Norm(const vector<double> &x) 
{
	return sqrt(DotProduct(x, x));
}
const vector<double> operator/(const vector<double> &x, const vector<double> &y) 
{
	vector<double> ret(x);
	for (int i = 0; i < (int) ret.size(); i++) ret[i] /= y[i];
	return ret;
}
const vector<double> operator+(const vector<double> &x, const vector<double> &y) 
{
	vector<double> ret(x);
	for (int i = 0; i < (int) ret.size(); i++) ret[i] += y[i];
	return ret;
}
const vector<double> operator-(const vector<double> &x, const vector<double> &y) 
{
	vector<double> ret(x);
	for (int i = 0; i < (int) ret.size(); i++) ret[i] -= y[i];
	return ret;
}
const vector<double> &operator+=(vector<double> &x, double c) 
{
	for (int i = 0; i < (int) x.size(); i++) x[i] += c;
	return x;
}
const vector<double> operator*(const vector<double> &x, double c) 
{
	vector<double> ret(x);
	for (int i = 0; i < (int) ret.size(); i++) ret[i] *= c;
	return ret;
}

