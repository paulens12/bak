#include <Constants.h>

#ifdef CUDA
#include "cuda_runtime.h"
#endif

// fp - f+1
// fm - f-1
// rp - r+1
// rm - r-1
// get next value of u
#ifdef CUDA
__device__
#endif
double getNextU(
	double u, double urp, double urm, double ufp, double ufm,
	double v, double vrp, double vrm, double vfp, double vfm,
	double ro, double df
)
{
	double ufp2 = (u + ufp) / 2;
	double ufm2 = (u + ufm) / 2;
	double urp2 = (u + urp) / 2;
	double urm2 = (u + urm) / 2;

	double rp2 = ro + dr / 2;
	double rm2 = ro - dr / 2;
	double rrff = ro * ro * df * df;

	double DuMember =
		(rp2 * (urp - u) - rm2 * (u - urm)) / (dr2 * ro) +
		(ufp - 2 * u + ufm) / rrff;
	double ChiMember =
		(rp2 * urp2 * (vrp - v) - rm2 * urm2 * (v - vrm)) / (ro * dr2) +
		(ufp2 * (vfp - v) - ufm2 * (v - vfm)) / rrff;
	double aMember = u * (1 - u);

	return (Du * DuMember - chi * ChiMember + au * aMember) * dt + u;
}

// get next value of v
#ifdef CUDA
__device__
#endif
double getNextV(
	double u,
	double v, double vrp, double vrm, double vfp, double vfm,
	double ro, double df
)
{
	double rp2 = ro + dr / 2;
	double rm2 = ro - dr / 2;
	double rrff = ro * ro * df * df;

	return (
		(rp2 * (vrp - v) - rm2 * (v - vrm)) / (ro * dr2) +
		(vfp - 2 * v + vfm) / rrff +
		u / (1 + Bv * u) - v
		) * dt + v;
}

#ifdef CUDA
__device__
#endif
void calcPoint(double* uOutput, double* vOutput, double* uInput, double* vInput, double ro, int r, int f) {
	double urp, urm, vrp, vrm, ufp, ufm, vfp, vfm, u, v;
	double df;
	int fp, fm;

	// sparse array towards the center
	if (r < R / 4 && f % 4 != 0)
		return;
	if (r < R / 2 && f % 2 != 0)
		return;

	u = uInput[F * r + f];
	v = vInput[F * r + f];

	urp = uInput[F * (r + 1) + f];
	vrp = vInput[F * (r + 1) + f];

	if (r == R / 4 && f % 4 == 2) {
		urm = (uInput[F * (r - 1) + f - 2] + uInput[F * (r - 1) + (f + 2) % F]) / 2;
		vrm = (vInput[F * (r - 1) + f - 2] + vInput[F * (r - 1) + (f + 2) % F]) / 2;
	}
	else if (r == R / 2 && f % 2 == 1) {
		urm = (uInput[F * (r - 1) + f - 1] + uInput[F * (r - 1) + (f + 1) % F]) / 2;
		vrm = (vInput[F * (r - 1) + f - 1] + vInput[F * (r - 1) + (f + 1) % F]) / 2;
	}
	else {
		urm = uInput[F * (r - 1) + f];
		vrm = vInput[F * (r - 1) + f];
	}

	// ufp, vfp, ufm, vfm
	if (r < R / 4) {
		// use every 4 cells
		fp = f + 4;
		fm = f - 4;
	}
	else if (r < R / 2) {
		// use every 2 cells
		fp = f + 2;
		fm = f - 2;
	}
	else {
		// use every cell
		fp = f + 1;
		fm = f - 1;
	}
	if (fp >= F) fp = 0;
	if (fm < 0) fm += F;
	ufp = uInput[F * r + fp];
	vfp = vInput[F * r + fp];
	ufm = uInput[F * r + fm];
	vfm = vInput[F * r + fm];

	// df
	if (r < R / 4)
		df = BASE_df * 4;
	else if (r < R / 2)
		df = BASE_df * 2;
	else
		df = BASE_df;

	uOutput[F * r + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, ro, df);
	vOutput[F * r + f] = getNextV(u, v, vrp, vrm, vfp, vfm, ro, df);
}

// apply boundary conditions
#ifdef CUDA
__device__
#endif
void calcBoundary(double* u, double* v, int thrn = 0, int thread_n = 1)
{
	for (int f = thrn; f < F; f += thread_n)
	{
		// no-flux boundary condition
		u[(R - 1) * F + f] = (4 * u[(R - 2) * F + f] - u[(R - 3) * F + f]) / 3;
		v[(R - 1) * F + f] = (4 * v[(R - 2) * F + f] - v[(R - 3) * F + f]) / 3;
	}

	for (int f = thrn * 4; f < F / 2; f += thread_n * 4)
	{
		// central symmetry boundary condition
		u[F / 2 + f] = u[f] = (u[F + f] + u[3 * F / 2 + f]) / 2;
		v[F / 2 + f] = v[f] = (v[F + f] + v[3 * F / 2 + f]) / 2;
	}
}
