#include <Constants.h>
#include "UV.h"

// get next value of u
double getNextU(double* u, double* v, double* o, int x, int y, int z)
{
	// u_(i_x+0.5,i_y,i_z)
	double uxp2 = (GET(u, x + 1, y, z) + GET(u, z, y, z)) / 2;
	// u_(i_x-0.5,i_y,i_z)
	double uxm2 = (GET(u, x - 1, y, z) + GET(u, z, y, z)) / 2;
	// ...
	double uyp2 = (GET(u, x, y + 1, z) + GET(u, z, y, z)) / 2;
	double uym2 = (GET(u, x, y - 1, z) + GET(u, z, y, z)) / 2;
	double uzp2 = (GET(u, x, y, z + 1) + GET(u, z, y, z)) / 2;
	double uzm2 = (GET(u, x, y, z - 1) + GET(u, z, y, z)) / 2;

	double DuMember =
		(GET(u, x + 1, y, z) - 2 * GET(u, x, y, z) + GET(u, x - 1, y, z)) / dx2 +
		(GET(u, x, y + 1, z) - 2 * GET(u, x, y, z) + GET(u, x, y - 1, z)) / dy2 +
		(GET(u, x, y, z + 1) - 2 * GET(u, x, y, z) + GET(u, x, y, z - 1)) / dz2_cube;
	double ChiMember =
		(uxp2 * (GET(v, x + 1, y, z) - GET(v, x, y, z)) - uxm2 * (GET(v, x, y, z) - GET(v, x - 1, y, z))) / dx2 +
		(uyp2 * (GET(v, x, y + 1, z) - GET(v, x, y, z)) - uym2 * (GET(v, x, y, z) - GET(v, x, y - 1, z))) / dy2 +
		(uzp2 * (GET(v, x, y, z + 1) - GET(v, x, y, z)) - uzm2 * (GET(v, x, y, z) - GET(v, x, y, z - 1))) / dz2_cube;
	double aMember = GET(u, x, y, z) * (1 - GET(u, x, y, z) / GET(o, x, y, z));

	return (Du * DuMember - chi * ChiMember + au * aMember) * dt + GET(u, x, y, z);
}

// get next value of v
double getNextV(double* u, double* v, int x, int y, int z)
{
	return (
		(GET(v, x + 1, y, z) - 2 * GET(v, x, y, z) + GET(v, x - 1, y, z)) / dx2 +
		(GET(v, x, y + 1, z) - 2 * GET(v, x, y, z) + GET(v, x, y - 1, z)) / dy2 +
		(GET(v, x, y, z + 1) - 2 * GET(v, x, y, z) + GET(v, x, y, z - 1)) / dz2_cube +
		GET(u, x, y, z) / (1 + Bv * GET(u, x, y, z)) - GET(v,x,y,z)
		) * dt + GET(v,x,y,z);
}

double getNextO(double* u, double* o, double o_z_plus_one, int x, int y, int z)
{
	return (
		Do * (
			(GET(o, x + 1, y, z) - 2 * GET(o, x, y, z) + GET(o, x - 1, y, z)) / dx2 +
			(GET(o, x, y + 1, z) - 2 * GET(o, x, y, z) + GET(o, x, y - 1, z)) / dy2 +
			(o_z_plus_one - 2 * GET(o, x, y, z) + GET(o, x, y, z - 1)) / dz2_cube
			) -
		gamma_o * GET(u, x, y, z)
		) * dt + GET(o, x, y, z);
}

void calcPoint(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput, int x, int y, int z) {
	GET(uOutput, x, y, z) = getNextU(uInput, vInput, oInput, x, y, z);
	GET(vOutput, x, y, z) = getNextV(uInput, vInput, x, y, z);
	GET(oOutput, x, y, z) = getNextO(uInput, oInput, GET(oInput, x, y, z + 1), x, y, z);
}

// apply boundary conditions
void calcBoundary(double* u, double* v, double* o, double* uPrev, double* oPrev, int thrn, int thread_n)
{
	for (int y = 1; y < Y - 1; y++) {
		for (int x = thrn + 1; x < X - 1; x += thread_n) {
			GET(u, x, y, 0) = fmax((4 * GET(u, x, y, 1) - GET(u, x, y, 2)) / 3, 0);
			GET(v, x, y, 0) = fmax((4 * GET(v, x, y, 1) - GET(v, x, y, 2)) / 3, 0);
			GET(o, x, y, 0) = fmax((4 * GET(o, x, y, 1) - GET(o, x, y, 2)) / 3, 0);
			GET(u, x, y, Z_cube - 1) = fmax((4 * GET(u, x, y, Z_cube - 2) - GET(u, x, y, Z_cube - 3)) / 3, 0);
			GET(v, x, y, Z_cube - 1) = fmax((4 * GET(v, x, y, Z_cube - 2) - GET(v, x, y, Z_cube - 3)) / 3, 0);
			GET(o, x, y, Z_cube - 1) = getNextO(uPrev, oPrev, o0, x, y, Z_cube - 1);
		}
		for (int z = thrn + 1; z < Z_cube - 1; z += thread_n) {
			GET(u, 0, y, z) = fmax((4 * GET(u, 1, y, z) - GET(u, 2, y, z)) / 3, 0);
			GET(v, 0, y, z) = fmax((4 * GET(v, 1, y, z) - GET(v, 2, y, z)) / 3, 0);
			GET(o, 0, y, z) = fmax((4 * GET(o, 1, y, z) - GET(o, 2, y, z)) / 3, 0);
			GET(u, X - 1, y, z) = fmax((4 * GET(u, X - 2, y, z) - GET(u, X - 3, y, z)) / 3, 0);
			GET(v, X - 1, y, z) = fmax((4 * GET(v, X - 2, y, z) - GET(v, X - 3, y, z)) / 3, 0);
			GET(o, X - 1, y, z) = fmax((4 * GET(o, X - 2, y, z) - GET(o, X - 3, y, z)) / 3, 0);
		}
	}
	for (int z = 1; z < Z_cube - 1; z++) {
		for (int x = thrn + 1; x < X - 1; x += thread_n) {
			GET(u, x, 0, z) = fmax((4 * GET(u, x, 1, z) - GET(u, x, 2, z)) / 3, 0);
			GET(v, x, 0, z) = fmax((4 * GET(v, x, 1, z) - GET(v, x, 2, z)) / 3, 0);
			GET(o, x, 0, z) = fmax((4 * GET(o, x, 1, z) - GET(o, x, 2, z)) / 3, 0);
			GET(u, x, Y - 1, z) = fmax((4 * GET(u, x, Y - 2, z) - GET(u, x, Y - 3, z)) / 3, 0);
			GET(v, x, Y - 1, z) = fmax((4 * GET(v, x, Y - 2, z) - GET(v, x, Y - 3, z)) / 3, 0);
			GET(o, x, Y - 1, z) = fmax((4 * GET(o, x, Y - 2, z) - GET(o, x, Y - 3, z)) / 3, 0);
		}
	}
}
