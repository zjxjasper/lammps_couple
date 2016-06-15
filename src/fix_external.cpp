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

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "update.h"
#include "fix_external.h"
#include "atom.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "comm.h"
#include "math.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{PF_CALLBACK,PF_ARRAY,PF_ESCM,PF_IVCDAMP};
enum{SCALAR,VECTOR};

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2
#define INVOKED_ARRAY 4



/* ---------------------------------------------------------------------- */

FixExternal::FixExternal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix external command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  me = comm->me;
  if (strcmp(arg[3],"pf/callback") == 0) {
    if (narg != 6) error->all(FLERR,"Illegal fix external command");
    mode = PF_CALLBACK;
    ncall = force->inumeric(FLERR,arg[4]);
    napply = force->inumeric(FLERR,arg[5]);
    if (ncall <= 0 || napply <= 0) 
      error->all(FLERR,"Illegal fix external command");
  } else if (strcmp(arg[3],"pf/array") == 0) {
    if (narg != 5) error->all(FLERR,"Illegal fix external command");
    mode = PF_ARRAY;
    napply = force->inumeric(FLERR,arg[4]);
    if (napply <= 0) error->all(FLERR,"Illegal fix external command");
  } else if(strcmp(arg[3],"pf/ESCM") == 0) {
    if (narg != 8) error->all(FLERR,"Illegal fix external command");
    mode = PF_ESCM;
    // arg [1]: ncall [2]: napply [3]:nnode_inter
    ncall = force->inumeric(FLERR,arg[4]);
    napply = force->inumeric(FLERR,arg[5]);
    nstore = force->inumeric(FLERR,arg[6]);
    nnode_inter = force->inumeric(FLERR,arg[7]);
    if(me == 0) printf("fix external/ESCM is called, ncall:%d, napply:%d, nstore:%d\n",ncall,napply,nstore);
  } else if(strcmp(arg[3],"pf/IVCdamp")== 0){
    if (narg != 9) error->all(FLERR,"Illegal fix external command");
    mode = PF_IVCDAMP;
    ncall = force->inumeric(FLERR,arg[4]);
    napply = force->inumeric(FLERR,arg[5]);
    nstore = force->inumeric(FLERR,arg[6])+ncall;
    nnode_inter = force->inumeric(FLERR,arg[7]);
    drag = force->numeric(FLERR,arg[8]);
    if(me == 0) printf("fix external/IVCdamp is called, ncall:%d, napply:%d, nstart :%d drag:%f\n",ncall,napply,nstore,drag);
  }
  else error->all(FLERR,"Illegal fix external command");

  callback = NULL;

  // perform initial allocation of atom-based array
  // register with Atom class
 
  fexternal = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);

  user_energy = 0.0;
  nevery = ncall;
//  if(me == 0) printf("mode is %d, ESCM mode is %d\n",mode,PF_ESCM);
  if(mode == PF_ESCM){
     flag_fs0_exist = 0;
     flag_fs0_stored = 0;
     int icompute;
     nvalues = nnode_inter*3;   // 1 or # of values of that compute?                
     int length;
     char* compute_id ="SVCforce" ;   // use preesure to test
     fs0 = NULL;
     memory->grow(fs0,nvalues,"external:fs0");
     for(int i = 0;i<nvalues;i++)
     {
//        if(me==0)  printf("initial value of fs0[%d] is %f\n",i,fs0[i]);
        // initialize fs0 value
        fs0[i] = 0.0;
     }
     flag_fs0_exist = 1;
     if(ncall > 0){
       icompute = modify->find_compute(compute_id);
       if(icompute==-1){
          error->warning(FLERR,"fix external ESCM mode without compute SVC/force");
          flag_SVC = 0;     
       }
       else{
          flag_SVC = 1;
          length = modify->compute[icompute]->size_vector;
          nrows = length;
          if(nrows!=nnode_inter*3){
             printf("nrows is %d \n",nrows);
            error->all(FLERR,"dof of compute SVC and fix_external ESCM mode not consistent");
          }
          column = new double[nrows];
          nvalid_call = nextvalid_call();
          modify->addstep_compute_all(nvalid_call);
        }
     }
//     if(me==0) printf("in create compute added at step %d\n",nvalid_call);
   } else if(mode = PF_IVCDAMP){
     nvalues = nnode_inter*3;   // 1 or # of values of that compute?                

     int length;
     char* compute_id ="CMdisp" ; 
     IVC_CMdisp = NULL;
     IVC_CMdisp0 = NULL;
     memory->grow(IVC_CMdisp,nvalues,"external:IVC_CMdisp");
     memory->grow(IVC_CMdisp0,nvalues,"external:IVC_CMdisp0");
     for(int i = 0;i<nvalues;i++)
     {
         IVC_CMdisp[i] = 0.0;
         IVC_CMdisp0[i] = 0.0;
     }
     flag_CMdisp_exist = 1;
     if(ncall > 0){
       icompute = modify->find_compute(compute_id);
       if(icompute==-1)  error->all(FLERR,"fix external ESCM mode does not have a valid compute IVC/CMV ID");
       length = modify->compute[icompute]->size_vector;
       nrows = length;
       if(nrows!=nnode_inter*3){
            printf("nrows is %d \n",nrows);
             error->all(FLERR,"dof of compute IVC and fix_external IVCCMV mode not consistent");
       }
       
       column = new double[nrows];
       nvalid_call = nextvalid_call();
       modify->addstep_compute_all(nvalid_call);
     }
   }
  
  //check created natoms_IVC,SVC


//  nvalid_apply = nextvalid_apply();
//  printf("nvalid is given as %d with ncall %d\n",nvalid,ncall);
}

/* ---------------------------------------------------------------------- */

FixExternal::~FixExternal()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,0);
  memory->destroy(fexternal);
  if(mode == PF_ESCM && flag_fs0_exist ==1){
     memory->destroy(fs0);
     delete [] column;
  }
  if((mode == PF_IVCDAMP)&& (flag_CMdisp_exist ==1)){
     memory->destroy(IVC_CMdisp);
     memory->destroy(IVC_CMdisp0);
     delete [] column;
  }
}

/* ---------------------------------------------------------------------- */

int FixExternal::setmask()
{
  int mask = 0;
  if (mode == PF_CALLBACK || mode == PF_ARRAY) {
    mask |= POST_FORCE;
    mask |= THERMO_ENERGY;
    mask |= MIN_POST_FORCE;
  }
  if (mode == PF_ESCM) {
    mask |= POST_FORCE;
    if(flag_SVC == 1){
       mask |= END_OF_STEP;
    }
  }
  if (mode == PF_IVCDAMP) {
    mask |= POST_FORCE;
    mask |= END_OF_STEP;
  }
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixExternal::init()
{
//  printf("fixexternal init called with mode being %d %d\n",mode,PF_ESCM); 
  if(mode == PF_ESCM){
     if(ncall > 0){
       if(flag_SVC == 1){
          char* compute_id ="SVCforce" ;   // use preesure to test
          icompute = modify->find_compute(compute_id);
   //     if(me==0)  printf("in init icompute is %d\n",icompute);
        // icompute is same value for same compute like c_p[1] c_p[2]
          if(icompute<0){
             error->all(FLERR,"Compute ID for fix external using SVC/force not found");
          }
          if(nvalid_call < update->ntimestep){
             nvalid_call = nextvalid_call();
             modify->addstep_compute_all(nvalid_call);
//            if(me==0) printf("in init compute added at step %d\n",nvalid_call);
          }
       }
     }
   }else if(mode == PF_IVCDAMP){
     if(ncall > 0){
       char* compute_id ="CMdisp" ;   // use preesure to test
       icompute = modify->find_compute(compute_id);
       if(icompute<0){
          error->all(FLERR,"Compute ID for fix external with name 'CMdisp' not found");
       }
       if(nvalid_call < update->ntimestep){
          nvalid_call = nextvalid_call();
          modify->addstep_compute_all(nvalid_call);
//         if(me==0) printf("in init compute added at step %d\n",nvalid_call);
       }
     }
   }
  if (mode == PF_CALLBACK && callback == NULL)
    error->all(FLERR,"Fix external callback function not set");
}

/* ---------------------------------------------------------------------- */

void FixExternal::setup(int vflag)
{
   bigint ntimestep = update->ntimestep; 
   post_force(vflag);
}

void FixExternal::end_of_step()
{
   bigint ntimestep = update->ntimestep; 
   int i;
//   if(me==0) printf("end of step is called at step %d with nvalid %d\n",ntimestep,nvalid_call);
//  be careful on nvalid should be different for end_of_step and post_force

   // at give timestep nstore,store calculated value into fs0
   if(ntimestep != nvalid_call ) return;
   if(mode == PF_ESCM){
     if(ntimestep == nstore){
//       if(me==0)  printf("f_surf stored at step %d,at nvalid?=%d,timestpe/ncall?=%d\n",ntimestep,(ntimestep==nvalid_call),(ntimestep%ncall==0));
       invoke_vector(ntimestep);
       for(i=0;i<nvalues;i++){
          fs0[i] = column[i];
        }
          if((comm->me==0)) printf("fs0 stored at time %d, fs0[0] is %f \n",ntimestep,fs0[0]);
        flag_fs0_stored = 1;
     }else if (ntimestep % ncall == 0){
//      if(me == 0)  printf("at step %d end of step is called\n",ntimestep);
        if(ncall > 0)     invoke_vector(ntimestep);
        if(flag_fs0_exist == 1)     distribute_force_SVC();
     }
   }else if(mode = PF_IVCDAMP){
     if(ntimestep == nstore){
       if(me==0)  printf("CMdisp stored at step %d,at nvalid?=%d,timestpe/ncall?=%d\n",ntimestep,(ntimestep==nvalid_call),(ntimestep%ncall==0));
       invoke_vector(ntimestep);
       for(i=0;i<nvalues;i++){
          IVC_CMdisp0[i] = column[i];
          IVC_CMdisp[i] = column[i];
        }
        if((me==0)) printf("storing disp0[0] is %f\n",i,IVC_CMdisp0[0]);
        flag_CMdisp_stored = 1;
     }else if (ntimestep % ncall == 0){
        if(ncall > 0)     invoke_vector(ntimestep);
        for(i=0;i<nvalues;i++){
          IVC_CMdisp0[i] = IVC_CMdisp[i];
          IVC_CMdisp[i] = column[i];
         }
        if(flag_CMdisp_exist == 1)     distribute_force_IVC();
     }
   }
}

void FixExternal::distribute_force_SVC()
{
   int i,j,k;
   int nlocal = atom->nlocal;
   bigint ntimestep = update->ntimestep; 
   int *natoms_SVC = domain->natoms_SVC;
   //diagnosis
   
   int flag_simple = 0;
   if(flag_simple == 1) return;
   for(i=0;i<nnode_inter;i++)
   {
//       printf("natoms_SVC[%d] is %d\n",i,natoms_SVC[i]);
      if(natoms_SVC[i]<=0){
        error->all(FLERR,"Fix external natoms_SVC is zero or negative");}
   }
   int *IVC_id = atom->IVC_id;
   int *tag = atom->tag;
   double **fext = atom->fext;
   int node_id;
   double weight;
   double df[3];
   /*** correct way is 
   delta fc = -(dt/tM)(fs(t) - fs(0))
   ***/
   double ratio = 0.001;  //corresponding value dt/TM, TM -> 50-500*dt
   int dof = 0;
   double diffmax[3];
   diffmax[0] = diffmax[1] = diffmax[2] = 0.0;
  double tmpd;
   for(j=0;j<nvalues;j++){
      tmpd = fs0[j];
//     if((me==0))  printf("fs0[%d] is %f\n",j,fs0[j]);
   } 
//   printf("check fs0 finished %d\n",me);

   int flag_distribute = 1;
   if(flag_distribute == 1){
   for(i=0;i<nlocal;i++){
      node_id = IVC_id[i];
      if(node_id<=0)  continue;
      if(natoms_SVC[node_id-1]<=0){
         printf("atom %d natoms belong to this node %d is non-positive %f\n",i,node_id,natoms_SVC[node_id-1]);
//         error->all(FLERR,"natoms_node is non-positive");
      }
      else{
         weight = 1.0/natoms_SVC[node_id-1];
         dof = node_id*3-3;
         if((dof<0)||(dof>nvalues-3)){
           printf("dof out of bound due to node_id %d, dof %d atom %d\n",node_id,dof,i);
           error->all(FLERR,"dof out of bound");
         }
         df[0]= (fs0[dof+0]-column[dof+0])*weight*ratio;
         df[1]= (fs0[dof+1]-column[dof+1])*weight*ratio;
         df[2]= (fs0[dof+2]-column[dof+2])*weight*ratio;
         fext[i][0] += df[0];
         fext[i][1] += df[1];
         fext[i][2] += df[2];
      }
   }}
//   printf("distribute force is done in proc %d\n",me);
}



void FixExternal::distribute_force_IVC()    // deal with distribute damping force
{
   int i,j,k,l;
   int node_id;
   int dof;
   double rmin,rmax,dr,r;
   rmin = 200.0-12.0;
   rmax = 200.0+12.0;
   int ilayer;    // # of layer atom sit in
   int nlayer = 6;  // total # of layers
   dr = (rmax-rmin)/nlayer;

   double df[3];
   double dt = update->dt;
   int *IVC_id = atom->IVC_id;
   int *tag = atom->tag;
   double **xa = atom->x;
   double **fext = atom->fext;
   int nlocal = atom->nlocal;
   double *v_CM = new double[nnode_inter*3];
   //drag coefficient in units metal, eV*s/m^2
   //if(me==0)  printf("drag is %f\n",drag);
   for(k=0;k<nnode_inter;k++){
      for(l=0;l<3;l++){
         dof = k*3+l;
         v_CM[dof] = (IVC_CMdisp[dof]-IVC_CMdisp0[dof])/(ncall*dt);
      }
//     if((me==0)&&(k<5))  printf(" v_CM[%d] is %f %f %f\n",k,v_CM[dof-2],v_CM[dof-1],v_CM[dof]);
   }
  
   for(i=0;i<nlocal;i++){
       node_id = -IVC_id[i];   // IVC id is -1 ~ -ninter, SVC id is 1 ~ ninter
       if(node_id<=0) continue;
       r = sqrt(xa[i][0]*xa[i][0]+xa[i][1]*xa[i][1]);
       ilayer = floor((r-rmin)/dr)+1;
       if(ilayer<0){ 
         ilayer = 0; }
       else if(ilayer>nlayer) {
           ilayer =nlayer; }
       for(j=0;j<3;j++){
          dof = node_id*3-3+j;
          df[j] = -v_CM[dof]*drag/nlayer*ilayer;
          fext[i][j] += df[j];
       }
//   if(i<5)      printf("in proc %d,node %d, atom %d, dof %d, v is %f, df is %f, fext is %f\n",me,node_id,i,dof-2,v_CM[dof-2],df[0],fext[i][0]);
   }     
   delete [] v_CM;
}

void FixExternal::invoke_vector(bigint ntimestep)
{
   int i,m;
//    printf("compute_vector is called at step %d in proc:%d\n",ntimestep,comm->me);
//   if(me==0) printf("invoke_vector is called at timestep %d, nvalues %d\n",ntimestep,nvalues);
   if(irepeat == 0)
      for(i=0;i<nvalues;i++)  column[i] = 0.0;

   modify->clearstep_compute();
   
//   icompute = modify->find_compute(compute_id);
   m = icompute;  
   Compute *compute = modify->compute[m];

   if(!(compute->invoked_flag & INVOKED_VECTOR)){
       compute->compute_vector();
       compute->invoked_flag |= INVOKED_VECTOR;
    }
    double *cvector = compute->vector;
//    if(me==0)  printf("check obtained vector from compute,%d %f %f\n",nrows,cvector[0],cvector[1]);
    for (i=0;i<nrows;i++){
        column[i] = cvector[i];
//       if(me==0)   printf("cvector[%d] is %f\n",i,cvector[i]);
    }
   irepeat ++;
   nvalid_call += ncall;    // call compute every ncall timestep 
   modify -> addstep_compute(nvalid_call);
//   if (me==0)   printf("by invoke_vector next nvalid_call is %d\n",nvalid_call);
//   int *natoms_SVC = domain->natoms_SVC;
}



/* ----------------------------------------------------------------------
   calculate nvalid = next step on which end_of_step does something
   can be this timestep if multiple of ncall
   startstep is lower bound on ncall multiple
------------------------------------------------------------------------- */

bigint FixExternal::nextvalid_call()
{
  bigint ntimestep = update->ntimestep;
  bigint nvalid_call;
  bigint startstep = 0;
  if(ntimestep <= nstore){
       nvalid_call = nstore;
//       if(me==0) printf("next nvalid is set as %d at timestep %d\n",nvalid_call,nstore);
  }
  else{
//     printf("WARNING: nstore is smaller than timestep when fix is applied\n");
     nvalid_call = update->ntimestep + ncall;
  }
  while (nvalid_call < startstep) nvalid_call += ncall;
//  if(me==0) printf("next valid_call is %d\n",nvalid_call);
  return nvalid_call;
}

bigint FixExternal::nextvalid_apply()
{
  bigint startstep = 0;
  bigint nvalid_apply = update->ntimestep + napply;
  while (nvalid_apply < startstep) nvalid_apply += napply;
  return nvalid_apply;
}
/* ---------------------------------------------------------------------- */

void FixExternal::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixExternal::post_force(int vflag)
{
  bigint ntimestep = update->ntimestep;
  int i,j,id;
  // invoke the callback in driver program
  // it will fill fexternal with forces

  if (mode == PF_CALLBACK && ntimestep % ncall == 0)
    (this->callback)(ptr_caller,update->ntimestep,
                     atom->nlocal,atom->tag,atom->x,fexternal);

  if(mode == PF_ARRAY && ntimestep % napply == 0) {
//    printf("apply is called in post_force at timestep %d\n",ntimestep);

    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int *group_tag = atom->IVC_id;
    
    for (i = 0; i < nlocal; i++)
    {
      if (mask[i] & groupbit) {
        f[i][0] += atom->fext[i][0];
        f[i][1] += atom->fext[i][1];
        f[i][2] += atom->fext[i][2];
      }
    }
  }
  // add forces from fexternal to atoms in group
  if(mode == PF_ESCM && ntimestep % napply == 0) {
//    printf("apply is called in post_force at timestep %d\n",ntimestep);
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int *group_tag = atom->IVC_id;

    for (i = 0; i < nlocal; i++)
    {
      if (mask[i] & groupbit) {
        f[i][0] += atom->fext[i][0];
        f[i][1] += atom->fext[i][1];
        f[i][2] += atom->fext[i][2];
      }
    }
  }

  if(mode == PF_IVCDAMP && ntimestep % napply == 0) {
//    printf("apply is called in post_force at timestep %d\n",ntimestep);
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int *group_tag = atom->IVC_id;

    for (i = 0; i < nlocal; i++)
    {
      if (mask[i] & groupbit) {
        f[i][0] += atom->fext[i][0];
        f[i][1] += atom->fext[i][1];
        f[i][2] += atom->fext[i][2];
      }
    }
    
    
  }



}

/* ---------------------------------------------------------------------- */

void FixExternal::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixExternal::set_energy(double eng)
{
  user_energy = eng;
}

/* ----------------------------------------------------------------------
   potential energy of added force
   up to user to set it via set_energy()
------------------------------------------------------------------------- */

double FixExternal::compute_scalar()
{
  return user_energy;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixExternal::memory_usage()
{
  double bytes = 3*atom->nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixExternal::grow_arrays(int nmax)
{
  memory->grow(fexternal,nmax,3,"external:fexternal");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixExternal::copy_arrays(int i, int j, int delflag)
{
  fexternal[j][0] = fexternal[i][0];
  fexternal[j][1] = fexternal[i][1];
  fexternal[j][2] = fexternal[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixExternal::pack_exchange(int i, double *buf)
{
  buf[0] = fexternal[i][0];
  buf[1] = fexternal[i][1];
  buf[2] = fexternal[i][2];
  return 3;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixExternal::unpack_exchange(int nlocal, double *buf)
{
  fexternal[nlocal][0] = buf[0];
  fexternal[nlocal][1] = buf[1];
  fexternal[nlocal][2] = buf[2];
  return 3;
}

/* ----------------------------------------------------------------------
   external caller sets a callback function to invoke in post_force()
------------------------------------------------------------------------- */

void FixExternal::set_callback(FnPtr caller_callback, void *caller_ptr)
{
  callback = caller_callback;
  ptr_caller = caller_ptr;
}


/**********************************************************************************


   here when ntimestep%ncall = 0, apply the surf compensation by calculate the fs from compute_SVC_force
  {
     double **fext = atom->fext;
     int *IVC_ID = atom->IVC_id;
     int *tag = atom->tag; //or tagint ?
      1. invoke compute_SVC to get the array A[3*nnode_inter] at ncall
         2. distribute A[3*nnode_inter] fext for each atom based on per atom group id(need weight!
         3. add fext on to f for all atoms (same as before)
         4. turn off the compensation by set ncall = 0
    
 
    1. invoke compute_SVC
     Compute *compute = modify ->compute[n];     


     for(i=0;i<nlocal;i++)
     {
       id = tag[i];         // id start from 0 or 1? use global id or local ID?
       if(id<0) error->all(FLERR,"Fix external can not find atom ID");
       IVC_id[id]
       if(){
         
       }
     }
  }
*************************************************************************************/
