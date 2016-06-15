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

/* ----------------------------------------------------------------------
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins U)
     K-space terms added by Stan Moore (BYU)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "string.h"
#include "compute_SVC_force.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "kspace.h"
#include "error.h"
#include "math.h"
#include "comm.h"
#include "math_const.h"
#include "stdio.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

ComputeSVCForce::ComputeSVCForce(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4){
     printf("narg is less than 4\n");
     error->all(FLERR,"Illegal compute SVC_force command");
  }

// e.g. compute 1 lower group/group upper pair yes
  nnode_inter = force->inumeric(FLERR,arg[3]);
  scalar_flag = vector_flag = 1;
  size_vector = nnode_inter*3;
  extscalar = 1;
  extvector = 1;

  if (nnode_inter<=0){
    error->all(FLERR,"Compute SVC nnode_interface needs to be positive integer");
  }

/*** no longer need to check group 2 and get jgroupbit
  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2,arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute group/group group ID does not exist");
  jgroupbit = group->bitmask[jgroup];
***/


  pairflag = 1;
  boundaryflag = 1;
  
  int iarg = 4;
  if(iarg < narg) error->all(FLERR,"Illegal compute SVC_force command");

  vector = new double[size_vector];
  domain->create_natoms_IVC(nnode_inter);
}

/* ---------------------------------------------------------------------- */

ComputeSVCForce::~ComputeSVCForce()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeSVCForce::init()
{
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0

  if (pairflag && force->pair == NULL)
    error->all(FLERR,"No pair style defined for compute SVC_force");
  if (force->pair_match("hybrid",0) == NULL && force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support compute SVC_force");


  if (pairflag) {
    pair = force->pair;
    cutsq = force->pair->cutsq;
  } else pair = NULL;


  // need an occasional half neighbor list

  if (pairflag) {
    int irequest = neighbor->request((void *) this);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeSVCForce::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

double ComputeSVCForce::compute_scalar()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  int i;
  for(i=0;i<size_vector;i++){
     vector[i] = 0.0;
  }

//  printf("pair_contribution is called in scalar\n");
  if (pairflag) pair_contribution();

  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeSVCForce::compute_vector()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  int i;
  for(i=0;i<size_vector;i++){
     vector[i] = 0.0;
  }
  pairflag = 1;
  if (pairflag) pair_contribution();
//  printf("new vector, %f %f %f\n",vector[0],vector[1],vector[2]);
//  for(i=0;i<size_vector;i++){
//     vector[i] = 0.01;
//  }
}

/* ---------------------------------------------------------------------- */

void ComputeSVCForce::pair_contribution()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,eng,fpair,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;
//  printf("pair_contribution is called\n");
  int *IVC_id = atom->IVC_id;
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  /**************
   groupbit and igroup already defined in compute.cpp
  igroup = group->find(arg[1]);
  if (igroup == -1) error->all(FLERR,"Could not find compute group ID");
  groupbit = group->bitmask[igroup];
  ***************/
  // invoke half neighbor list (will copy or build if necessary)
  neighbor->build_one(list->index);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I,J are not in 2 groups

  int n,m;
  int IVC_typei,IVC_typej;
  n = size_vector;

  

  double *one = new double[n+1];
  // notice here one[0] no longer means energy since we calculate multi group-group interfaction together
  for(m=0;m<n+1;m++){
     one[m] = 0.0;
  }
  
//  printf("before loop\n");
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
//    printf("atomcalled, ii %d ilist[ii] %d,group 1->cont %d,IVC_id %d \n",ii,i,(mask[i]&groupbit),IVC_id[i]);
    if ((mask[i] & groupbit)) continue; // skip if atom I is inside(not IVC or SVC)
    if (IVC_id[i] <= 0) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    IVC_typei = IVC_id[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if(IVC_id[j] >= 0) continue;	// double check if atom J neither IVC or SVC, skip it
       
      IVC_typej = IVC_id[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        eng = pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);

        // energy only computed once so tally full amount
        // force tally is jgroup acting on igroup

//          one[0] += eng;
         one[IVC_typei*3-2] += delx*fpair;
         one[IVC_typei*3-1] += dely*fpair;
         one[IVC_typei*3-0] += delz*fpair;

        // energy computed twice so tally half amount
        // only tally force if I own igroup atom

      }
//    printf("in SVC_force, one[0] : %f, one[3] %f\n",one[0],one[3]);
    }
//    printf("ii %d i %d,IVC_type %d, fpair %f,one %f\n",ii,i,IVC_typei,fpair,one[IVC_typei*3-2]);
  }
  double *all = new double[n+1];
  MPI_Allreduce(one,all,n+1,MPI_DOUBLE,MPI_SUM,world);
  scalar += all[0];
  for(m=0;m<n;m++){
     vector[m] += all[m+1];
//     printf("SVC computed force vector[%d] is %f\n",m+1,vector[m]);
  }
//  printf("computed force is %f %f %f\n",one[1],all[1],all[11]);
  delete [] one;
  delete [] all;
}

/* ---------------------------------------------------------------------- */

