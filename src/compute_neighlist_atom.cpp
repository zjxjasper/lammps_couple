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
   Contributing author: Wan Liang (Chinese Academy of Sciences)
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "compute_neighlist_atom.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math.h"

using namespace LAMMPS_NS;

#define MAXNEAR 16
enum{NEIGHLIST,DIST0};
/* ---------------------------------------------------------------------- */

ComputeNeighlistAtom::ComputeNeighlistAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute neighlist/atom command");

  peratom_flag = 1;
  size_peratom_cols = MAXNEAR;

  double cutoff = force->numeric(FLERR,arg[3]);
  if (cutoff < 0.0) error->all(FLERR,"Illegal compute neighlist/atom command");
  cutsq = cutoff*cutoff;
  if (strcmp(arg[4],"neighlist") == 0){
    option = NEIGHLIST;
    size_peratom_cols = MAXNEAR;
  }
  else if (strcmp(arg[4],"dist0") == 0){
    option = DIST0;
  size_peratom_cols = MAXNEAR*4;
  }else{
      error->all(FLERR,"Illegal compute neighlist/atom option");
  }


  nmax = 0;
  nearest = NULL;
  nnearest = NULL;
  neighborlist = NULL;
  if(option==DIST0) dist0 = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeNeighlistAtom::~ComputeNeighlistAtom()
{
  if(nearest != NULL)   memory->destroy(nearest);
  if(nnearest != NULL)    memory->destroy(nnearest);
  memory->destroy(neighborlist);
  if(option == DIST0)    memory->destroy(dist0);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighlistAtom::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Compute neighlist/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,"Compute neighlist/atom cutoff is longer than pairwise cutoff");

  // cannot use neighbor->cutneighmax b/c neighbor has not yet been init

  if (2.0*sqrt(cutsq) > force->pair->cutforce + neighbor->skin &&
      comm->me == 0)
    error->warning(FLERR,"Compute neighlist/atom cutoff may be too large to find "
                   "ghost atom neighbors");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"neighlist/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute neighlist/atom defined");

  // need an occasional full neighbor list

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeNeighlistAtom::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNeighlistAtom::compute_peratom()
{
  int i,j,k,ii,jj,kk,m,n,inum,jnum,ncount;
  int inttemp;
  double doubletemp;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;

  invoked_peratom = update->ntimestep;

  // grow arrays if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(nearest);
    memory->destroy(nnearest);
    memory->destroy(neighborlist);
    if(option == DIST0)    memory->destroy(dist0);
//    memory->destroy(dist0);
    
    nmax = atom->nmax;

    memory->create(nearest,nmax,MAXNEAR,"neighlist:nearest");
    memory->create(nnearest,nmax,"neighlist:nnearest");
    memory->create(neighborlist,nmax,MAXNEAR,"neighlist:neighborlist");
    if(option == NEIGHLIST){
      array_atom = neighborlist;
    }
    else if(option == DIST0){
      memory->create(dist0,nmax,MAXNEAR*4,"neighlist:dist0");
       array_atom = dist0;
    }
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list->index);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // find the neigbours of each atom within cutoff using full neighbor list
  // nearest[] = atom indices of nearest neighbors, up to MAXNEAR
  // do this for all atoms, not just compute group
  // since CNA calculation requires neighbors of neighbors

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int nerror = 0;
  //printf("inum is %d \n",inum);
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
	//printf(" ii %d, i %d, tag[i] %d, tag[ii] %d, jlist %d, jum %d \n",ii,i,atom->tag[i],atom->tag[ii],jlist[0],jnum);
    n = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
	 // printf(" jj %d, j %d \n", jj, j);
      j &= NEIGHMASK;

      delx = -xtmp + x[j][0];
      dely = -ytmp + x[j][1];
      delz = -ztmp + x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < cutsq) {
        if (n < MAXNEAR){
           nearest[i][n] = j;
           if(option==DIST0){
              dist0[i][n*4+1] = delx;
              dist0[i][n*4+2] = dely;
              dist0[i][n*4+3] = delz;
           }           
           n = n+1;   
	}
        else {
          nerror++;
          break;
        }
      }
    }
    nnearest[i] = n;
  }

  for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
    if ((mask[i] & groupbit)) {
	for(ncount=0;ncount<MAXNEAR;ncount++)
 	{
	//printf("nnearest %d, nearest[j] %d, tag(nearest[j]) %d \n",nnearest[i],nearest[i][ncount],atom->tag[nearest[i][ncount]]);
          if(ncount<nnearest[i])
	  {
	     inttemp = atom->tag[nearest[i][ncount]];
	     doubletemp = double(inttemp);
	     neighborlist[i][ncount] = doubletemp;
             if(option == DIST0){
                dist0[i][ncount*4+0] = doubletemp;
             }
	//   printf("ncount %d, tag[i] %d, tag[ii] %d, tag[neighborlist] %f \n",ncount, atom->tag[i], atom->tag[ii], doubletemp);
	  }
	  else{
              neighborlist[i][ncount] = -1.0; 
              if(option==DIST0){
                 dist0[i][ncount*4+0] = -1.0;
                 dist0[i][ncount*4+1] = 0.0;
                 dist0[i][ncount*4+2] = 0.0;
                 dist0[i][ncount*4+3] = 0.0;
              }
	  }
	}
     }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeNeighlistAtom::memory_usage()
{
  double bytes;
  if(option == NEIGHLIST){
     bytes = nmax * MAXNEAR * sizeof(double);
  }
  else if(option == DIST0){
     bytes = nmax * MAXNEAR * 4 *sizeof(double);
  } 
  return bytes;
}
