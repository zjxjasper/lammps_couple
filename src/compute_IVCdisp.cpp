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

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "compute_IVCdisp.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "fix_store.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeIVCDisp::ComputeIVCDisp(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal computes IVCdisp command");
  
  me = comm->me;
  nnode_inter = force->inumeric(FLERR,arg[3]);
  scalar_flag = vector_flag = 1;
  size_vector = nnode_inter*3;
  extscalar = 1;
  extvector = 1;

  if(nnode_inter<=0){
    error->all(FLERR,"Compute IVCdisp nnode_interface needs to be positive integer");
  }

  // create a new fix STORE style
  // id = compute-ID + COMPUTE_STORE, fix group = compute group

  int n = strlen(id) + strlen("_COMPUTE_STORE") + 1;
  id_fix = new char[n];
  strcpy(id_fix,id);
  strcat(id_fix,"_COMPUTE_STORE");

  char **newarg = new char*[5];
  newarg[0] = id_fix;
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "STORE";
  newarg[3] = (char *) "1";
  newarg[4] = (char *) "3";
  modify->add_fix(5,newarg);
  fix = (FixStore *) modify->fix[modify->nfix-1];
  delete [] newarg;

  // calculate xu,yu,zu for fix store array
  // skip if reset from restart file

  if (fix->restart_reset) fix->restart_reset = 0;
  else {
    double **xoriginal = fix->astore;

    double **x = atom->x;
    int *mask = atom->mask;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) domain->unmap(x[i],image[i],xoriginal[i]);
      else xoriginal[i][0] = xoriginal[i][1] = xoriginal[i][2] = 0.0;
  }

  // IVCdisplacement array

  nmax = 0; 

  vector = new double[size_vector];
  domain->create_natoms_IVC(nnode_inter);
//  printf("in proc %d IVCdisp is created \n",comm->me);
  
}

/* ---------------------------------------------------------------------- */

ComputeIVCDisp::~ComputeIVCDisp()
{
  // check nfix in case all fixes have already been deleted

  if (modify->nfix) modify->delete_fix(id_fix);

  delete [] id_fix;
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeIVCDisp::init()
{
  // set fix which stores original atom coords

  int ifix = modify->find_fix(id_fix);
  if (ifix < 0) error->all(FLERR,"Could not find compute IVCdisp fix ID");
  fix = (FixStore *) modify->fix[ifix];
//  printf("in proc %d IVCdisp is inited \n",comm->me);
}

/* ---------------------------------------------------------------------- */


void ComputeIVCDisp::reduce_disp()
{
  invoked_peratom = update->ntimestep;

  // grow local displacement array if necessary

  if (atom->nlocal > nmax) {
    nmax = atom->nmax;
  }

  // dx,dy,dz = displacement of atom from original position
  // original unwrapped position is stored by fix
  // for triclinic, need to unwrap current atom coord via h matrix

  double **xoriginal = fix->astore;
  int i;
  /**************
  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  **************/

  double **x = atom->x;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  MPI_Barrier(world); 
  double *h = domain->h;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  int xbox,ybox,zbox;
  double dx,dy,dz;
   

  int *natoms_IVC = domain->natoms_IVC;
  int *IVC_id = atom->IVC_id;
  int index_inter;

  int n,m;
  n = size_vector;
  if(size_vector != nnode_inter*3) error->all(FLERR,"nnode_inter and size_vector not consistent");
  double *one = new double[n+1];
  int *onecount = new int[nnode_inter];
  for(m=0;m<n+1;m++){
     one[m] = 0.0;
  }
  for(m=0;m<nnode_inter;m++){
     onecount[m] = 0;
  }

  if (domain->triclinic == 0) {
    for (i = 0; i < nlocal; i++)
    {
      if (mask[i] & groupbit) {
        index_inter = -IVC_id[i];
        if(index_inter<=0) continue;
        xbox = (image[i] & IMGMASK) - IMGMAX;
        ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        zbox = (image[i] >> IMG2BITS) - IMGMAX;
        dx = x[i][0] + xbox*xprd - xoriginal[i][0];
        dy = x[i][1] + ybox*yprd - xoriginal[i][1];
        dz = x[i][2] + zbox*zprd - xoriginal[i][2];
 
        onecount[index_inter-1]  += 1;
        one[index_inter*3-2] += dx;
        one[index_inter*3-1] += dy;
        one[index_inter*3-0] += dz;
//        printf("atom %d,dx %f one %f,index_inter %d,count %d, proc:%d\n",i,dx,one[index_inter*3-2],index_inter,onecount[index_inter-1],comm->me);
      }
    }
//  printf("before allreduce one to all %f %f %f proc %d\n",one[1],one[2],one[n+1],comm->me);
   } else {
    for (i = 0; i< nlocal; i++){
      if (mask[i] & groupbit) {
        index_inter = -IVC_id[i];
        if(index_inter<=0) continue;
        xbox = (image[i] & IMGMASK) - IMGMAX;
        ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        zbox = (image[i] >> IMG2BITS) - IMGMAX;
        dx = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox - xoriginal[i][0];
        dy = x[i][1] + h[1]*ybox + h[3]*zbox - xoriginal[i][1];
        dz = x[i][2] + h[2]*zbox - xoriginal[i][2];

        onecount[index_inter-1]  += 1;
        one[index_inter*3-3] += dx;
        one[index_inter*3-2] += dy;
        one[index_inter*3-1] += dz;

      }
    }
  }
  double *all = new double[n+1];
  int *allcount = new int[nnode_inter];
  for(m=0;m<nnode_inter;m++){
     allcount[m] = 0;
//     printf("before allreduce onecount[%d]=%d, proc:%d \n",m,onecount[m],comm->me);
  }
  MPI_Allreduce(one,all,n+1,MPI_DOUBLE,MPI_SUM,world); 
  MPI_Allreduce(onecount,allcount,nnode_inter,MPI_INT,MPI_SUM,world);
//  for(m=0;m<nnode_inter;m++){
//    printf("after allreduce all[%d]=%f %f %f, proc:%d \n",m,all[m*3-3+1],all[m*3-3+2],all[m*3-3+3],comm->me);
//  }
  
  for(m=0;m<nnode_inter;m++){
    if(allcount[m]!=natoms_IVC[m]){
       printf("natoms_inter[%d] is %d, correct value is %d, by proc %d\n",m,allcount[m],natoms_IVC[m],me);
       error->all(FLERR,"Could not find compute displace/atom fix ID");
    }
  }

  scalar  = 0.0;
  for(m=0;m<nnode_inter;m++){
    vector[3*m+0] = all[3*m+1]/allcount[m];
    vector[3*m+1] = all[3*m+2]/allcount[m];
    vector[3*m+2] = all[3*m+3]/allcount[m];
//    printf("all is %f %f %f, count is %d, result is %f %f %f\n",all[3*m+1],all[3*m+2],all[3*m+2],vector[3*m+0],vector[3*m+1],vector[3*m+2]);
  }

  delete [] one;
  delete [] all;
  delete [] onecount;
  delete [] allcount;



}


double ComputeIVCDisp::compute_scalar()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  int i;
  for(i=0;i<size_vector;i++){
     vector[i] = 0.0;
  }

  reduce_disp();
  return scalar;

}

void ComputeIVCDisp::compute_vector()
{
   invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  int i;
  for(i=0;i<size_vector;i++){
     vector[i] = 0.0;
  }
  reduce_disp();
//  for(i=0;i<size_vector;i++){
//     printf("sizevector %d, in proc %d, compute_vector[%d] is %f\n",size_vector,me,i,vector[i]);
//  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeIVCDisp::memory_usage()
{
  double bytes = size_vector*sizeof(double);
  return bytes;
}
