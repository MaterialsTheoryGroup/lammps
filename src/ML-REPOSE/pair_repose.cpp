/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#include "pair_repose.h"

#include "librepose.hpp"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRepose::PairRepose(LAMMPS *lmp) : Pair(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

PairRepose::~PairRepose()
{
}

/* ---------------------------------------------------------------------- */

void PairRepose::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // ----- prepare input in repose format -----

  int num_ions = atom->nlocal;

  char ion_names[6*atom->nlocal];
  for (int i=0; i<6*atom->nlocal; ++i)
    ion_names[i] = '\0';
  #pragma omp parallel for
  for (int ii=0; ii<num_ions; ++ii) {
    int i = list->ilist[ii];
    auto ion_name = element_names[atom->type[i]-1];
    std::copy(ion_name.begin(), ion_name.end(), &ion_names[6*i]);
  }

  double ion_positions[3*atom->nlocal];
  #pragma omp parallel for
  for (int ii=0; ii<num_ions; ++ii) {
    int i = list->ilist[ii];
    ion_positions[3*i+0] = atom->x[i][0];
    ion_positions[3*i+1] = atom->x[i][1];
    ion_positions[3*i+2] = atom->x[i][2];
  }

  double lattice_car[9];
  lattice_car[0] = domain->h[0];
  lattice_car[1] = 0.0;
  lattice_car[2] = 0.0;
  lattice_car[3] = domain->h[5];
  lattice_car[4] = domain->h[1];
  lattice_car[5] = 0.0;
  lattice_car[6] = domain->h[4];
  lattice_car[7] = domain->h[3];
  lattice_car[8] = domain->h[2];

  double e;
  double f[3*num_ions];
  double s[9];

  // ----- evaluate potential -----

  c_eval_ddp(num_ions, ion_names, ion_positions, lattice_car, &e, f, s);

  // ----- extract output -----

  // energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    eng_vdwl = e;
    // todo!
    //#pragma omp parallel for reduction(+:eng_vdwl)
    //for (int ii=0; ii<list->inum; ++ii) {
    //  int i = list->ilist[ii];
    //  eng_vdwl +=
    //}
  }

  // forces
  //   -> derivatives of total energy
  #pragma omp parallel for
  for (int ii=0; ii<list->inum; ++ii) {
    int i = list->ilist[ii];
    atom->f[i][0] = f[3*i+0];
    atom->f[i][1] = f[3*i+1];
    atom->f[i][2] = f[3*i+2];
  }

  // todo possibly!
  // site energies
  //   -> local atoms only
  //if (eflag_atom) {
  //  #pragma omp parallel for
  //  for (int ii=0; ii<list->inum; ++ii) {
  //    int i = list->ilist[ii];
  //    eatom[i] = 
  //  }
  //}

  // repose virials
  //   -> derivatives of sum of site energies of local atoms
  if (vflag_global) {
    virial[0] = s[0];
    virial[1] = s[4];
    virial[2] = s[8];
    virial[3] = 0.5*(s[5] + s[7]);
    virial[4] = 0.5*(s[2] + s[6]);
    virial[5] = 0.5*(s[1] + s[3]);
  }

  // repose site virials not available
  if (vflag_atom) {
    error->all(FLERR, "ERROR: pair_repose does not support vflag_atom.");
  }
}

/* ---------------------------------------------------------------------- */

void PairRepose::settings(int narg, char **arg)
{
  if (narg > 0) {
    error->all(FLERR, "Too many pair_style arguments for pair_style repose.");
  }
}

/* ---------------------------------------------------------------------- */

void PairRepose::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  char seedname[80];
  std::copy(&arg[2][0], &arg[2][std::strlen(arg[2])], std::begin(seedname));
  std::cout << "PairRepose.coeff:  setting seedname to " << seedname << "." << std::endl;
  c_set_seedname(seedname);
  std::cout << "PairRepose.coeff:  loading ddp." << std::endl;
  c_read_ddp();
  rcut = c_ddp_rcut();
  std::cout << "PairRepose.coeff:  ddp rcut is " << rcut << "." << std::endl;

  // extract atomic numbers from pair_coeff
  std::cout << "  - The pair_coeff element names are:";
  for (int i=3; i<narg; ++i) {
    std::cout << " " << arg[i];
    element_names.push_back(arg[i]);
  }
  std::cout << std::endl;

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

void PairRepose::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style repose requires newton pair on.");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairRepose::init_one(int i, int j)
{
  return c_ddp_rcut();
}

void PairRepose::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}
