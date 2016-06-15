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

ComputeStyle(deformgrad/atom,ComputeDeformGradAtom)

#else

#ifndef LMP_COMPUTE_DEFORMGRAD_ATOM_H
#define LMP_COMPUTE_DEFORMGRAD_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeDeformGradAtom : public Compute {
 public:
  ComputeDeformGradAtom(class LAMMPS *, int, char **);
  ~ComputeDeformGradAtom();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int me;
  int option;
  int nmax;
  double cutsq;
  char *id_fix1,*id_fix2;
  class FixStoreState *fix1;
  class FixStoreState *fix2;
  double **strain,**deformgrad;
  void matrix3_multi(double **,double **,double **);
  void matrix3_inv(double **,double **);
  void reform_matrix(double **,double *,int,int);
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
