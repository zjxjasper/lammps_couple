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

#ifdef FIX_CLASS

FixStyle(external,FixExternal)

#else

#ifndef LMP_FIX_EXTERNAL_H
#define LMP_FIX_EXTERNAL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixExternal : public Fix {
 public:
  double **fexternal;

  FixExternal(class LAMMPS *, int, char **);
  ~FixExternal();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void end_of_step();
  void min_post_force(int);
  double compute_scalar();
  double compute_vector();

  void set_energy(double eng);

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  typedef void (*FnPtr)(void *, bigint, int, tagint *, double **, double **);
  void set_callback(FnPtr, void *);

 private:
  int mode,ncall,napply,nstore,nstart;
  int icompute;
  int nvalues,me;
  int flag_fs0_exist,flag_fs0_stored;
  int flag_CMdisp_exist,flag_CMdisp_stored;
  int nrows;
  int nrepeat,irepeat,nfreq;
  int nnode_inter;
  double *vector;
  double *column;
  double *fs0;
  double drag;
  double *IVC_CMdisp,*IVC_CMdisp0;
  bigint nvalid_apply,nvalid_call;

  FnPtr callback;
  int flag_SVC;
  void *ptr_caller;
  double user_energy;
  void invoke_vector(bigint);
  void allocate_values(int);
  void distribute_force_IVC();    // for apply damping based on CMV of IVC, used in mode pf/IVCDAMP
  void distribute_force_SVC();    // for ghost-force dynamic correction, used in mode pf/ESCM
  bigint nextvalid_call();
  bigint nextvalid_apply();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix external callback function not set

This must be done by an external program in order to use this fix.

*/
