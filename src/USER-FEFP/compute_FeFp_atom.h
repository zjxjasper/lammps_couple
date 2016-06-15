/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(fefp/atom,ComputeFeFpAtom)

#else

#ifndef LMP_COMPUTE_FEFP_ATOM_H
#define LMP_COMPUTE_FEFP_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeFeFpAtom : public Compute {
 public:
  ComputeFeFpAtom(class LAMMPS *, int, char **);
  ~ComputeFeFpAtom();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int me;
  double xprd0,yprd0,zprd0;
  int nneigh;           //  # of nearest neighbors, 12 for fcc hcp, 8 for bcc
  int option;
  int nmax;
  double cutsq;         // cutoff to identify the nearest neighbors, 0.5*(1/sqrt(2)+1)*a0 for fcc if want nearest neighbor
  double slip_cut;      // larger than which is identified as bond displacement caused by plastic deformation
  char *id_fix1,*id_compute1;
  char *id_fix2;
  int flag_ave;
  int tstore;           // which timestep the reference configuration is defined
  class FixStoreState *fix1;
//  class FixAveAtom *fix2;
  class Fix *fix2;
  class ComputeNeighlistAtom *compute1;
  double **strain,**deformgrad,**FeFp;
  void matrix3_multi(double **,double **,double **);
  void matrix3_inv(double **,double **,int *);
  void reform_matrix(double **,double *,int,int);
  void Unitify(double **,int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Could not find compute displace/atom fix ID

Self-explanatory.

*/
