    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_wilson.cc

    Copyright (C) 2015

Author: Richard Rollins <rprollins@users.noreply.github.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */

#include <Grid.h>

// Class for the dynamical evolution of cosmological domain walls

class Walls
{

public:

  // Vectorized scalar and vectorized scalar field data types
  typedef Grid::iScalar<Grid::vRealD> vScalar;
  typedef Grid::Lattice<vScalar>      vScalarField;

  // Real value data type
  typedef Grid::RealD RealType;

  Walls(Grid::GridCartesian * Grid, vScalarField * phi1, vScalarField * phi2) :

    stencil(Grid,npoint,Even,directions,displacements),
    dnu(dlnnu * nu0),
    iz(a0 / (1.0 - a0)),
    omega0(0.5 * W0 * nu0),
    V0(0.5 * M_PI * M_PI / ( omega0 * omega0)),
    vidx2(1.0 / (nu0 * nu0)),
    vdnu(dnu),
    vidnu(1.0 / dnu),
    vqmiu(0.0)

  {
    // Initialize pointers to scalar fields
    phi_old = phi2;
    phi_now = phi1;
    phi_new = phi2;

    // Initialize time varaibles
    nu_old = nu0;
    nu_now = nu0;
  }

  void timestep()
  {
    // Calculate global varaibles
    delta      = 0.5 * alpha * iz * dnu / nu_now;
    v1mdelta   = 1.0 - delta;
    v4V0atbeta = 4.0 * V0 * pow( pow(nu_old,iz), beta );
    vi1pdelta  = 1.0 / (1.0 + delta);

    // Iterate over all grid sites
    PARALLEL_FOR_LOOP
    for(int i=0; i<phi_now->_grid->oSites(); i++)
    {
      // Laplacian as grid site i
      Lphi = Laplacian(i);
      // Current time derivative at i
      dphi_now = (phi_now->_odata[i] - phi_old->_odata[i]) * vidnu;
      // Potential gradient at i
      dVdphi = v4V0atbeta * ( phi_now->_odata[i]*( phi_now->_odata[i]*phi_now->_odata[i] - 1.0 ) + vqmiu );
      // New time derivative at i
      dphi_new = ( v1mdelta*dphi_now + vdnu*( Lphi - dVdphi ) ) * vi1pdelta;
      // Dynamical evolution of phi at i
      phi_new->_odata[i] = phi_now->_odata[i] + vdnu * dphi_new;
    }

    // Halo exchange
    stencil.HaloExchange(*phi_new,compressor);

    // Increment time
    nu_old  = nu_now;
    nu_now += dnu;

    // Switch array aliases
    phi_old = const_cast<vScalarField const *>(phi_now);
    phi_now = const_cast<vScalarField const *>(phi_new);
    phi_new = const_cast<vScalarField *>(phi_old);
  }

private:

  // Stencil variables

  const int ndim   = 3;
  const int npoint = 6;
  const int Even   = 0;
  const std::vector<int> directions    = std::vector<int>({ 0, 1, 2, 0, 1, 2});
  const std::vector<int> displacements = std::vector<int>({ 1, 1, 1,-1,-1,-1});

  Grid::CartesianStencil<vScalar,vScalar> stencil;
  Grid::SimpleCompressor<vScalar> compressor;

  // Simulation constants and variables

  const RealType alpha = 3.0;       // Constant alpha
  const RealType beta  = 0.0;       // Constant beta
  const RealType W0    = 10.0;      // Ratio of wall thickness to horizon size at nu0
  const RealType nu0   = 1.0;       // Initial conformal time
  const RealType a0    = 0.6666666; // Scale factor at nu0
  const RealType dlnnu = 0.25;      // Delta (ln conformal time) per timestep

  const RealType iz;     // Inverse redshift
  const RealType omega0; // Wall thickness
  const RealType V0;     // Potential barrier
  RealType nu_now;       // Current conformal time
  RealType nu_old;       // Previous conformal time

  const RealType dnu;   // Conformal timestep size
  const vScalar  vdnu;  // Vector conformal timestep size
  const vScalar  vidnu; // Vector inverse conformal timestep size
  const vScalar  vidx2; // Vector inverse of the squared grid size
  const vScalar  vqmiu; // Vector constant term in potential gradient

  RealType delta;      // Repeated term in dynamical evolution equation
  vScalar  v1mdelta;   // Vector 1 - delta
  vScalar  vi1pdelta;  // Vector 1 / ( 1 + delta )
  vScalar  v4V0atbeta; // Vector potential gradient factor

  vScalar Lphi;     // Laplacian of phi
  vScalar dphi_now; // Current time derivative of phi
  vScalar dphi_new; // New time derivative of phi
  vScalar dVdphi;   // Potential gradient

  vScalarField const * phi_old; // Scalar field phi at previous timestep
  vScalarField const * phi_now; // Scalar field phi at current timestep
  vScalarField *       phi_new; // Write location for phi at next timestep

  inline vScalar Laplacian(int i)
  {
    // Variables used by stencil
    Grid::StencilEntry *SE;
    vScalar SV;
    int permute_type;

    // Return value
    vScalar Lphi;

    // Calculate Laplacian in 3 dimensions from 6-point stencil

    // Loop over stencil entries
    for(int j=0; j<2*ndim; j++)
    {
      SE = stencil.GetEntry(permute_type,j,i);
      if ( SE->_is_local && SE->_permute )
        permute(SV,phi_now->_odata[SE->_offset],permute_type);
      else if (SE->_is_local)
        SV = phi_now->_odata[SE->_offset];
      else
        SV = stencil.comm_buf[SE->_offset];
      if(j==0) {Lphi  = SV;}
      else     {Lphi += SV;}
    }

    // Complete Laplacian
    Lphi -= 6.0 * phi_now->_odata[i];
    return Lphi * vidx2;
  }
};

int main (int argc, char ** argv)
{
  // Initialize grid library
  Grid::Grid_init(&argc,&argv);

  // Benchmark constants
  const int Nloop = 10000; // Benchmark iterations
  const int Nd    = 3;     // Three spatial dimensions

  // Lattice, MPI and SIMD layouts and OpenMP threads
  std::vector<int> latt_size   = Grid::GridDefaultLatt();
  std::vector<int> mpi_layout  = Grid::GridDefaultMpi();
  std::vector<int> simd_layout = Grid::GridDefaultSimd(Nd,Grid::vRealD::Nsimd());
  int threads = Grid::GridThread::GetThreads();
  int vol     = latt_size[0]*latt_size[1]*latt_size[2];

  // Cartesian grid
  Grid::GridCartesian Grid(latt_size,simd_layout,mpi_layout);

  // Random number genereator
  Grid::GridParallelRNG pRNG(&Grid);
  pRNG.SeedRandomDevice();

  // Scalar fields phi1 and phi2 are initialized to the same sample
  // from the uniform random field [-1.0,1.0] on the Grid
  Walls::vScalarField phi1(&Grid);
  Walls::vScalarField phi2(&Grid);
  random(pRNG,phi1);
  phi1 = (2.*phi1) - 1.;
  phi2 = phi1;

  // Walls object
  Walls walls(&Grid,&phi1,&phi2);

  // Start timer
  double start = Grid::usecond();

  // Loop over timesteps
  for(int ii=0; ii<Nloop; ii++) { walls.timestep(); }

  // Total run time
  double stop = Grid::usecond();
  double time = (stop-start) * 1.0E-6;

  // Memory throughput and Flop rate
  double bytes = vol * sizeof(Walls::RealType) * 2 * Nloop;
  double flops = double(vol) * double(22) * double(Nloop);
  std::cout << Grid::GridLogMessage << std::setprecision(5)
            << "GB/s: "    << 1.0E-9 * bytes / time << "\t"
            << "GFlop/s: " << 1.0E-9 * flops / time << std::endl;

  // Finalize
  Grid::Grid_finalize();
  return(0);
}
