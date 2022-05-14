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
	double u, double urp, double urm, double ufp, double ufm, double uzp, double uzm,
	double v, double vrp, double vrm, double vfp, double vfm, double vzp, double vzm,
	double o, double ro, double df
)
{
	double ufp2 = (u + ufp) / 2;
	double ufm2 = (u + ufm) / 2;
	double urp2 = (u + urp) / 2;
	double urm2 = (u + urm) / 2;
	double uzp2 = (u + uzp) / 2;
	double uzm2 = (u + uzm) / 2;

	double rp2 = ro + dr / 2;
	double rm2 = ro - dr / 2;
	double rrff = ro * ro * df * df;

	double DuMember =
		(rp2 * (urp - u) - rm2 * (u - urm)) / (dr2 * ro) +
		(ufp - 2 * u + ufm) / rrff +
		(uzp - 2 * u + uzm) / dz2;
	double ChiMember =
		(rp2 * urp2 * (vrp - v) - rm2 * urm2 * (v - vrm)) / (ro * dr2) +
		(ufp2 * (vfp - v) - ufm2 * (v - vfm)) / rrff +
		(uzp2 * (vzp - v) - uzm2 * (v - vzm)) / dz2;
	double aMember = u * (1 - u / o);

	return (Du * DuMember - chi * ChiMember + au * aMember) * dt + u;
}

// get next value of v
#ifdef CUDA
__device__
#endif
double getNextV(
	double u,
	double v, double vrp, double vrm, double vfp, double vfm, double vzp, double vzm,
	double ro, double df
)
{
	double rp2 = ro + dr / 2;
	double rm2 = ro - dr / 2;
	double rrff = ro * ro * df * df;

	return (
		(rp2 * (vrp - v) - rm2 * (v - vrm)) / (ro * dr2) +
		(vfp - 2 * v + vfm) / rrff +
		(vzp - 2 * v + vzm) / dz2 +
		u / (1 + Bv * u) - v
		) * dt + v;
}

#ifdef CUDA
__device__
#endif
double getNextO(
	double o, double orp, double orm, double ofp, double ofm, double ozp, double ozm,
	double u, double ro, double df
)
{
	double rp2 = ro + dr / 2;
	double rm2 = ro - dr / 2;
	double rrff = ro * ro * df * df;
	return (
		Do * (
			(rp2 * (orp - o) - rm2 * (o - orm)) / (dr2 * ro) +
			(ofp - 2 * o + ofm) / rrff +
			(ozp - 2 * o + ozm) / dz2
			) - gamma_o * u
		) * dt + o;
}

#ifdef CUDA
__device__
#endif
double calcBoundaryO(double* oInput, double u, double ro, int r, int f) {
	// sparse array towards the center
	if (r < R / 4 && f % 4 != 0)
		return 0.0;
	if (r < R / 2 && f % 2 != 0)
		return 0.0;
	double orm, df;
	int fp, fm;

	if (r == R / 4 && f % 4 == 2) {
		orm = (oInput[R * F * (Z - 1) + F * (r - 1) + f - 2] + oInput[R * F * (Z - 1) + F * (r - 1) + (f + 2) % F]) / 2;
	}
	else if (r == R / 2 && f % 2 == 1) {
		orm = (oInput[R * F * (Z - 1) + F * (r - 1) + f - 1] + oInput[R * F * (Z - 1) + F * (r - 1) + (f + 1) % F]) / 2;
	}
	else {
		orm = oInput[R * F * (Z - 1) + F * (r - 1) + f];
	}

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

	if (r < R / 4)
		df = BASE_df * 4;
	else if (r < R / 2)
		df = BASE_df * 2;
	else
		df = BASE_df;

	return getNextO(
		oInput[R * F * (Z - 1) + F * r + f],
		oInput[R * F * (Z - 1) + F * (r + 1) + f],
		orm,
		oInput[R * F * (Z - 1) + F * r + fp],
		oInput[R * F * (Z - 1) + F * r + fm],
		o0,
		oInput[R * F * (Z - 2) + F * r + f], u, ro, df);
}

#ifdef CUDA
__device__
#endif
void calcPoint(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput, double ro, int r, int f, int z) {
	double urp, urm, vrp, vrm, orp, orm, ufp, ufm, vfp, vfm, ofp, ofm, u, v, o;
	double df;
	int fp, fm;

	// sparse array towards the center
	if (r < R / 4 && f % 4 != 0)
		return;
	if (r < R / 2 && f % 2 != 0)
		return;

	u = uInput[R * F * z + F * r + f];
	v = vInput[R * F * z + F * r + f];
	o = oInput[R * F * z + F * r + f];

	// rp
	urp = uInput[R * F * z + F * (r + 1) + f];
	vrp = vInput[R * F * z + F * (r + 1) + f];
	orp = oInput[R * F * z + F * (r + 1) + f];

	// rm
	if (r == R / 4 && f % 4 == 2) {
		urm = (uInput[R * F * z + F * (r - 1) + f - 2] + uInput[R * F * z + F * (r - 1) + (f + 2) % F]) / 2;
		vrm = (vInput[R * F * z + F * (r - 1) + f - 2] + vInput[R * F * z + F * (r - 1) + (f + 2) % F]) / 2;
		orm = (oInput[R * F * z + F * (r - 1) + f - 2] + oInput[R * F * z + F * (r - 1) + (f + 2) % F]) / 2;
	}
	else if (r == R / 2 && f % 2 == 1) {
		urm = (uInput[R * F * z + F * (r - 1) + f - 1] + uInput[R * F * z + F * (r - 1) + (f + 1) % F]) / 2;
		vrm = (vInput[R * F * z + F * (r - 1) + f - 1] + vInput[R * F * z + F * (r - 1) + (f + 1) % F]) / 2;
		orm = (oInput[R * F * z + F * (r - 1) + f - 1] + oInput[R * F * z + F * (r - 1) + (f + 1) % F]) / 2;
	}
	else {
		urm = uInput[R * F * z + F * (r - 1) + f];
		vrm = vInput[R * F * z + F * (r - 1) + f];
		orm = oInput[R * F * z + F * (r - 1) + f];
	}

	// fp, fm
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
	ufp = uInput[R * F * z + F * r + fp];
	vfp = vInput[R * F * z + F * r + fp];
	ofp = oInput[R * F * z + F * r + fp];
	ufm = uInput[R * F * z + F * r + fm];
	vfm = vInput[R * F * z + F * r + fm];
	ofm = oInput[R * F * z + F * r + fm];

	// zp, zm
	double uzp = uInput[R * F * (z + 1) + F * r + f];
	double vzp = vInput[R * F * (z + 1) + F * r + f];
	double ozp = oInput[R * F * (z + 1) + F * r + f];
	double uzm = uInput[R * F * (z - 1) + F * r + f];
	double vzm = vInput[R * F * (z - 1) + F * r + f];
	double ozm = oInput[R * F * (z - 1) + F * r + f];

	// df
	if (r < R / 4)
		df = BASE_df * 4;
	else if (r < R / 2)
		df = BASE_df * 2;
	else
		df = BASE_df;

	uOutput[R * F * z + F * r + f] = getNextU(u, urp, urm, ufp, ufm, uzp, uzm, v, vrp, vrm, vfp, vfm, vzp, vzm, o, ro, df);
	vOutput[R * F * z + F * r + f] = getNextV(u, v, vrp, vrm, vfp, vfm, vzp, vzm, ro, df);
	oOutput[R * F * z + F * r + f] = getNextO(o, orp, orm, ofp, ofm, ozp, ozm, u, ro, df);
}

#ifdef CUDA
__device__
#endif
void calcTopBottom(double* u, double* v, double* o, double ro, int r, int f)
{
	u[r * F + f] = (4 * u[R * F + r * F + f] - u[2 * R * F + r * F + f]) / 3;
	v[r * F + f] = (4 * v[R * F + r * F + f] - v[2 * R * F + r * F + f]) / 3;
	o[r * F + f] = (4 * o[R * F + r * F + f] - o[2 * R * F + r * F + f]) / 3;

	u[(Z - 1) * R * F + r * F + f] = (4 * u[(Z - 2) * R * F + r * F + f] - u[(Z - 3) * R * F + r * F + f]) / 3;
	v[(Z - 1) * R * F + r * F + f] = (4 * v[(Z - 2) * R * F + r * F + f] - v[(Z - 3) * R * F + r * F + f]) / 3;
}

// apply boundary conditions
#ifdef CUDA
__device__
#endif
void calcBoundary(double* u, double* v, double* o, int thrn_z = 0, int thread_n_z = 1, int thrn_f = 0, int thread_n_f = 1)
{
	for (int z = thrn_z; z < Z; z += thread_n_z) {
		for (int f = thrn_f; f < F; f += thread_n_f)
		{
			// no-flux boundary condition
			u[R * F * z + (R - 1) * F + f] = (4 * u[R * F * z + (R - 2) * F + f] - u[R * F * z + (R - 3) * F + f]) / 3;
			v[R * F * z + (R - 1) * F + f] = (4 * v[R * F * z + (R - 2) * F + f] - v[R * F * z + (R - 3) * F + f]) / 3;
			o[R * F * z + (R - 1) * F + f] = (4 * o[R * F * z + (R - 2) * F + f] - o[R * F * z + (R - 3) * F + f]) / 3;
		}

		for (int f = thrn_f * 4; f < F / 2; f += thread_n_f * 4)
		{
			// central symmetry boundary condition
			u[R * F * z + F / 2 + f] = u[R * F * z + f] = (u[R * F * z + F + f] + u[R * F * z + 3 * F / 2 + f]) / 2;
			v[R * F * z + F / 2 + f] = v[R * F * z + f] = (v[R * F * z + F + f] + v[R * F * z + 3 * F / 2 + f]) / 2;
			o[R * F * z + F / 2 + f] = o[R * F * z + f] = (o[R * F * z + F + f] + o[R * F * z + 3 * F / 2 + f]) / 2;
		}
	}

	for (int r = thrn_z + 1; r < R / 4; r += thread_n_z) {
		double ro = r * dr;
		for (int f = thrn_f * 4; f < F; f += thread_n_f * 4) {
			calcTopBottom(u, v, o, ro, r, f);
		}
	}

	for (int r = thrn_z + R / 4; r < R / 2; r += thread_n_z) {
		double ro = r * dr;
		for (int f = thrn_f * 2; f < F; f += thread_n_f * 2) {
			calcTopBottom(u, v, o, ro, r, f);
		}
	}

	for (int r = thrn_z + R / 2; r < R - 1; r += thread_n_z) {
		double ro = r * dr;
		for (int f = thrn_f; f < F; f += thread_n_f) {
			calcTopBottom(u, v, o, ro, r, f);
		}
	}
}
