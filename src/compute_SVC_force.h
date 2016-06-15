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

ComputeStyle(SVC/force,ComputeSVCForce)

#else

#ifndef LMP_COMPUTE_SVC_FORCE_H
#define LMP_COMPUTE_SVC_FORCE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSVCForce : public Compute {
 public:
  ComputeSVCForce(class LAMMPS *, int, char **);
  ~ComputeSVCForce();
  void init();
  void init_list(int, class NeighList *);
  double compute_scalar();
  void compute_vector();

 private:
  double **cutsq;
  int nnode_inter;
  int pairflag,boundaryflag;
  class Pair *pair;
  class NeighList *list;
  
  void pair_contribution();
};

}

#endif
#endif

