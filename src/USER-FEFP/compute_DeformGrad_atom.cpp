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
#include "compute_DeformGrad_atom.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "fix_store_state.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
using namespace LAMMPS_NS;

enum{DEFORMGRAD,STRAIN};
/* ---------------------------------------------------------------------- */

ComputeDeformGradAtom::ComputeDeformGradAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal computes DeformGradAtom command");
  
  me = comm->me;
  peratom_flag = 1;

  double cutoff = force->numeric(FLERR,arg[3]);
  if (strcmp(arg[4],"deformgrad") == 0){
    option = DEFORMGRAD;
  size_peratom_cols = 9;
  }
  else if (strcmp(arg[4],"strain") == 0){
    option = STRAIN;
  size_peratom_cols = 6;
  }else{
      error->all(FLERR,"Illegal compute DeformGrad/atom option");
  }

  if(cutoff < 0.0) error->all(FLERR,"Illegal compute DeformGrad/atom cutoff");
  cutsq = cutoff*cutoff;

  // create a new fix STORE style
  // id = compute-ID + COMPUTE_STORE, fix group = compute group
  // group has to be 'all'
  //
  int n1 = strlen(id) + strlen("_COMPUTE_STORE1") + 1;
  id_fix1 = new char[n1];
  strcpy(id_fix1,id);
  strcat(id_fix1,"_COMPUTE_STORE1");

  char **newarg1 = new char*[40];
  newarg1[0] = id_fix1;
  newarg1[1] = group->names[igroup];
  newarg1[2] = (char *) "store/state";
  newarg1[3] = (char *) "0";
  newarg1[4] = (char *) "c_Neigh1[1]";
  newarg1[5] = (char *) "c_Neigh1[2]";
  newarg1[6] = (char *) "c_Neigh1[3]";
  newarg1[7] = (char *) "c_Neigh1[4]";
  newarg1[8] = (char *) "c_Neigh1[5]";
  newarg1[9] = (char *) "c_Neigh1[6]";
  newarg1[10] = (char *) "c_Neigh1[7]";
  newarg1[11] = (char *) "c_Neigh1[8]";
  newarg1[12] = (char *) "c_Neigh1[9]";
  newarg1[13] = (char *) "c_Neigh1[10]";
  newarg1[14] = (char *) "c_Neigh1[11]";
  newarg1[15] = (char *) "c_Neigh1[12]";
  newarg1[16] = (char *) "c_Neigh1[13]";
  newarg1[17] = (char *) "c_Neigh1[14]";
  newarg1[18] = (char *) "c_Neigh1[15]";
  newarg1[19] = (char *) "c_Neigh1[16]";
  newarg1[20] = (char *) "c_Neigh1[17]";
  newarg1[21] = (char *) "c_Neigh1[18]";
  newarg1[22] = (char *) "c_Neigh1[19]";
  newarg1[23] = (char *) "c_Neigh1[20]";
  newarg1[24] = (char *) "c_Neigh1[21]";
  newarg1[25] = (char *) "c_Neigh1[22]";
  newarg1[26] = (char *) "c_Neigh1[23]";
  newarg1[27] = (char *) "c_Neigh1[24]";
  newarg1[28] = (char *) "c_Neigh1[25]";
  newarg1[29] = (char *) "c_Neigh1[26]";
  newarg1[30] = (char *) "c_Neigh1[27]";
  newarg1[31] = (char *) "c_Neigh1[28]";
  newarg1[32] = (char *) "c_Neigh1[29]";
  newarg1[33] = (char *) "c_Neigh1[30]";
  newarg1[34] = (char *) "c_Neigh1[31]";
  newarg1[35] = (char *) "c_Neigh1[32]";
  newarg1[36] = (char *) "c_Neigh1[33]";
  newarg1[37] = (char *) "c_Neigh1[34]";
  newarg1[38] = (char *) "c_Neigh1[35]";
  newarg1[39] = (char *) "c_Neigh1[36]";

  modify->add_fix(40,newarg1);
  fix1 = (FixStoreState *) modify->fix[modify->nfix-1];
  delete [] newarg1;


  
  int n2 = strlen(id) + strlen("_COMPUTE_STORE2") + 1;
  id_fix2 = new char[n2];
  strcpy(id_fix2,id);
  strcat(id_fix2,"_COMPUTE_STORE2");

  char **newarg2 = new char*[16];
  newarg2[0] = id_fix2;
  newarg2[1] = group->names[igroup];
  newarg2[2] = (char *) "store/state";
  newarg2[3] = (char *) "0";
  newarg2[4] = (char *) "c_Neigh2[1]";
  newarg2[5] = (char *) "c_Neigh2[2]";
  newarg2[6] = (char *) "c_Neigh2[3]";
  newarg2[7] = (char *) "c_Neigh2[4]";
  newarg2[8] = (char *) "c_Neigh2[5]";
  newarg2[9] = (char *) "c_Neigh2[6]";
  newarg2[10] = (char *) "c_Neigh2[7]";
  newarg2[11] = (char *) "c_Neigh2[8]";
  newarg2[12] = (char *) "c_Neigh2[9]";
  newarg2[13] = (char *) "c_Neigh2[10]";
  newarg2[14] = (char *) "c_Neigh2[11]";
  newarg2[15] = (char *) "c_Neigh2[12]";
  modify->add_fix(16,newarg2);
  fix2 = (FixStoreState *) modify->fix[modify->nfix-1];
  delete [] newarg2;


  nmax = 0; 
  deformgrad = NULL;
  strain = NULL;
  
}

/* ---------------------------------------------------------------------- */

ComputeDeformGradAtom::~ComputeDeformGradAtom()
{
  // check nfix in case all fixes have already been deleted

  if (modify->nfix) modify->delete_fix(id_fix1);
  if (modify->nfix) modify->delete_fix(id_fix2);
  delete [] id_fix1;
  delete [] id_fix2;
  if(option == DEFORMGRAD){
    memory->destroy(deformgrad);
  }else if(option == STRAIN){
    memory->destroy(strain);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeDeformGradAtom::init()
{
  // set fix which stores original atom coords

  int ifix1 = modify->find_fix(id_fix1);
  if (ifix1 < 0) error->all(FLERR,"Could not find compute DeformGrad/atom fix ID for store x0");
  fix1 = (FixStoreState *) modify->fix[ifix1];

  int ifix2 = modify->find_fix(id_fix2);
  if (ifix2 < 0) error->all(FLERR,"Could not find compute DeformGrad/atom fix ID for store neighlist");
  fix2 = (FixStoreState *) modify->fix[ifix2];
}

/* ---------------------------------------------------------------------- */


void ComputeDeformGradAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  int nlocal = atom->nlocal;
  int i,j;

  if (atom->nlocal > nmax) {
    if(option == DEFORMGRAD){
      memory->destroy(deformgrad);    
      nmax = atom->nmax;
      memory->create(deformgrad,nmax,9,"DeformGrad/atom:deformgrad");
      array_atom = deformgrad;
    }
    else if(option == STRAIN){
      memory->destroy(deformgrad);    
      memory->destroy(strain);    
      nmax = atom->nmax;
      memory->create(deformgrad,nmax,9,"DeformGrad/atom:deformgrad");
      memory->create(strain,nmax,6,"DeformGrad/atom:strain");
      array_atom = strain;
    }
  }
  double **cdist0 = fix1->array_atom;
  double **neighlist0 = fix2->array_atom;

  if(option==DEFORMGRAD){
    for(i=0;i<nlocal;i++){
       for(j=0;j<9;j++){
         deformgrad[i][j] = 0.0;
       }
    }
  }else if(option==STRAIN){
    for(i=0;i<nlocal;i++){
       for(j=0;j<6;j++){
         strain[i][j] = 0.0;
       }
    }
  }
   
  double **xa = atom->x;
  int n_id,n_id_global;
  double dist0[3],distT[3],xc[3],xn[3];
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double rcheck;
  double **eta,**inveta,**omega,**Flocal;
  double candi_dist;
  double Elocal[3][3];
  int m,n,l;


  memory->create(eta,3,3,"DeformGrad/atom:eta");
  memory->create(Flocal,3,3,"DeformGrad/atom:Flocal");
  memory->create(inveta,3,3,"DeformGrad/atom:inveta");
  memory->create(omega,3,3,"DeformGrad/atom:omega");

//  printf("check first, xa[0] %f %f %f, xa[1987] %f %f %f, xa[1988] %f %f %f\n",xa[0][0],xa[0][1],xa[0][2],xa[1987][0],xa[1987][1],xa[1987][2],xa[1988][0],xa[1988][1],xa[1988][2]);

//  printf("check first, xa[0] %f %f %f, xa[1987] %f %f %f, xa[1988] %f %f %f\n",xa[0][0],xa[0][1],xa[0][2],xa[1628][0],xa[1628][1],xa[1628][2],xa[1538][0],xa[1538][1],xa[1538][2]);
  for(i=0;i<nlocal;i++){
     for(m=0;m<3;m++){
        for(n=0;n<3;n++){
            eta[m][n] = omega[m][n] = 0.0;}}


     xc[0] = xa[i][0];
     xc[1] = xa[i][1];
     xc[2] = xa[i][2];
     for(j=0;j<12;j++){
        n_id_global = (int)(neighlist0[i][j]);

        if(n_id_global<0) continue;
        n_id = atom->map(n_id_global);
//        n_id = n_id_global;
        dist0[0] = cdist0[i][j*3];
        dist0[1] = cdist0[i][j*3+1];
        dist0[2] = cdist0[i][j*3+2];
        xn[0] = xa[n_id][0] ;
        xn[1] = xa[n_id][1] ;
        xn[2] = xa[n_id][2] ;
        distT[0] = xn[0] - xc[0];
        if(distT[0]>0.5*xprd){
           distT[0] = distT[0]-xprd*round(distT[0]/xprd);
        }else if(distT[0]<-0.5*xprd){
           distT[0] = distT[0]-xprd*round(distT[0]/xprd);
        }
        distT[1] = xn[1] - xc[1];
        if(distT[1]>0.5*yprd){
           distT[1] = distT[1]-yprd*round(distT[1]/yprd);
        }else if(distT[1]<-0.5*yprd){
           distT[1] = distT[1]-yprd*round(distT[1]/yprd);
        }
        distT[2] = xn[2] - xc[2];
        if(distT[2]>0.5*zprd){
           distT[2] = distT[2]-zprd*round(distT[2]/zprd);
        }else if(distT[2]<-0.5*zprd){
           distT[2] = distT[2]-zprd*round(distT[2]/zprd);
        }
        
//        printf("neighlist global and local id is %d %d, distT %f %f %f\n",(int)(neighlist0[i][j]),n_id,distT[0],distT[1],distT[2]);
//        if(i<10)    printf("atom %d, neigh %d dist0 %f %f %f xa %f %f %f, xa' %f %f %f\n",i,j,xc[0],xc[1],xc[2],xn[0],xn[1],xn[2],xa[n_id][0],xa[n_id][1],xa[n_id][2]);
        for(m=0;m<3;m++){
           for(n=0;n<3;n++){
              eta[m][n] += dist0[m]*dist0[n];
              omega[m][n] += distT[m]*dist0[n];
           }
        }
//        printf("checkpoint after calc eta and omega %f %f %f %f\n",eta[0][0],eta[1][1],omega[0][0],omega[1][1]);
     }
     matrix3_inv(eta,inveta);
     matrix3_multi(omega,inveta,Flocal);
     for(m=0;m<3;m++){
       for(n=0;n<3;n++){
         deformgrad[i][m*3+n] = -Flocal[m][n];                // somehow the sign is wrong, here jsut add -1 for simplicity, should find the error somewhere in compute_neighlist or this file
         Elocal[m][n] = 0.0;
         for(l=0;l<3;l++){
            Elocal[m][n] += (Flocal[l][m]*Flocal[l][n])/2;
         }
         if(m==n) Elocal[m][n] -= 0.5;
       }
     }
     if(option == STRAIN){
       strain[i][0] = Elocal[0][0];
       strain[i][1] = Elocal[1][1];
       strain[i][2] = Elocal[2][2];
       strain[i][3] = Elocal[0][1];
       strain[i][4] = Elocal[0][2];
       strain[i][5] = Elocal[1][2];
     }
  }
     

  memory->destroy(eta);
  memory->destroy(inveta);
  memory->destroy(Flocal);
  memory->destroy(omega);
}

void ComputeDeformGradAtom::matrix3_inv(double **A,double **B)
{
    double det = A[0][0]*A[1][1]*A[2][2]+A[0][1]*A[1][2]*A[2][0]+A[0][2]*A[1][0]*A[2][1]-A[0][0]*A[1][2]*A[2][1]-A[0][1]*A[1][0]*A[2][2]+A[0][2]*A[1][1]*A[2][0];
    if(det == 0) error->all(FLERR,"in compute deformgrad/atom, Determint is 0");
    else{
        double invdet = 1/det;
        B[0][0] = (A[1][1]*A[2][2]-A[1][2]*A[2][1])*invdet;
        B[0][1] = (A[1][0]*A[2][2]-A[1][2]*A[2][0])*invdet;
        B[0][2] = (A[1][0]*A[2][1]-A[1][1]*A[2][0])*invdet;
        B[1][0] = (A[0][1]*A[2][2]-A[0][2]*A[2][1])*invdet;
        B[1][1] = (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invdet;
        B[1][2] = (A[0][0]*A[2][1]-A[0][1]*A[2][0])*invdet;
        B[2][0] = (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invdet;
        B[2][1] = (A[0][0]*A[1][2]-A[0][2]*A[1][0])*invdet;
        B[2][2] = (A[0][0]*A[1][1]-A[0][1]*A[1][0])*invdet;
    }
   
}

void ComputeDeformGradAtom::matrix3_multi(double **A,double **B,double **C)
{
    for(int m=0;m<3;m++){
       for(int n=0;n<3;n++){
          C[m][n] = 0.0;
          for(int l=0;l<3;l++){
             C[m][n]+= A[m][l]*B[l][n];
          }
       }
    }
}

void ComputeDeformGradAtom::reform_matrix(double **A,double *b,int m,int n)
{
   int dof;
   for(int i=0;i<m;i++){
      for(int j=0;j<n;j++){
          dof = i*n+j;
          printf("in reform i %d, j %d, dof %d A[i][j] %f b[dof]%f\n",i,j,dof,A[i][j],b[dof]);
          b[dof] = A[i][j];
      }
   }
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeDeformGradAtom::memory_usage()
{
  double bytes = nmax*9*sizeof(double);
  return bytes;
}
