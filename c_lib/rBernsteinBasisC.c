/*==========================================================
 * Bernstein
 *========================================================*/
#include <stdio.h>
#include <stdlib.h>
#include "tIGA.h"
#include <math.h>
/* The computational routine */
double factorial0(int n)
{
    int i;
    double t1 = n;
    if (n==0 ){
        return 1;
    }
    for (i=n-1;i>0;i--){
        t1*= i;
    }
    return t1;
}

int BernsteinBasis0(int n, double *uuu, double *wt, double *N, double *dNx, double *dNy, double *ddNx, double *ddNy, double *ddNxy)
{
    int i, j, k, idx;
    int num = (n+1)*(n+2)/2;
    double u = uuu[0];
    double v = uuu[1];
    double w = uuu[2];
    double W = 0.0;
    double dWx = 0.0;
    double dWy = 0.0;
	double ddWx = 0.0;
    double ddWy = 0.0;
	double ddWxy = 0.0;
    double c, x, y, z, xx, yy, zz, xy, xz, yz;
//     N      = (double *)malloc(sizeof(double)*num);
// 	dN = init2DArray(num, 2);
    
    for (k = 0; k <= n; ++k){
    for (j = 0; j <= n-k; ++j){ //21 entries for n = 5
        i = n-k-j;
        idx = (k+1)+(2*n-i+3)*i/2-1;
        c = factorial0(n)/factorial0(i)/factorial0(j)/factorial0(k);
        N[idx] = c*pow(u,i)*pow(v,j)*pow(w,k);
        if (i == 0) {
			x = 0;
			xx = 0;
		}
        else {
           x = i*pow(u,i-1)*pow(v,j)*pow(w,k);
		   if (i == 1) 
			   xx = 0;
		   else 
			   xx = i*(i-1)*pow(u,i-2)*pow(v,j)*pow(w,k);
        }
		
        if (j == 0) {
			y = 0;
			yy = 0;
		}
        else {
           y = j*pow(u,i)*pow(v,j-1)*pow(w,k);
		   if (j == 1) 
			   yy = 0;
		   else 
			   yy = j*(j-1)*pow(u,i)*pow(v,j-2)*pow(w,k);
        }
		
        if (k == 0) {
			z = 0;
			zz = 0;
		}
        else {
           z = k*pow(u,i)*pow(v,j)*pow(w,k-1);
		   if (k == 1) 
			   zz = 0;
		   else 
			   zz = k*(k-1)*pow(u,i)*pow(v,j)*pow(w,k-2);
        }
		
		if ((i == 0)||(j == 0))
			xy = 0;
		else 
			xy = i*j*pow(u,i-1)*pow(v,j-1)*pow(w,k);
		
		if ((i == 0)||(k == 0))
			xz = 0;
		else 
			xz = i*k*pow(u,i-1)*pow(v,j)*pow(w,k-1);
		
		if ((j == 0)||(k == 0))
			yz = 0;
		else 
			yz = j*k*pow(u,i)*pow(v,j-1)*pow(w,k-1);
		
        dNx[idx] = c*(x-z);
        dNy[idx] = c*(y-z);
		ddNx[idx] = c*(xx+zz-2*xz);
		ddNy[idx] = c*(yy+zz-2*yz);
		ddNxy[idx] = c*(xy+zz-xz-yz);
    }
    }    
    
    // rational basis and derivatives
    for (i = 0; i < num; ++i){
    W   = W + N[i]*wt[i];
    dWx = dWx + dNx[i]*wt[i];
    dWy = dWy + dNy[i]*wt[i];
	ddWx = ddWx + ddNx[i]*wt[i];
    ddWy = ddWy + ddNy[i]*wt[i];
    ddWxy = ddWxy + ddNxy[i]*wt[i];
    }
    
    for (i = 0; i < num; ++i){

	ddNx[i] = wt[i]/pow(W,2)*(dWx*dNx[i]+ddNx[i]*W) - wt[i]/pow(W,2)*(dNx[i]*dWx+N[i]*ddWx) - 2*dWx*wt[i]/pow(W,3)*(W*dNx[i]-N[i]*dWx);
	ddNy[i] = wt[i]/pow(W,2)*(dWy*dNy[i]+ddNy[i]*W) - wt[i]/pow(W,2)*(dNy[i]*dWy+N[i]*ddWy) - 2*dWy*wt[i]/pow(W,3)*(W*dNy[i]-N[i]*dWy);	
	ddNxy[i] = wt[i]/pow(W,2)*(dWy*dNx[i]+ddNxy[i]*W) - wt[i]/pow(W,2)*(dNy[i]*dWx+N[i]*ddWxy) - 2*dWy*wt[i]/pow(W,3)*(W*dNx[i]-N[i]*dWx);	
	
    dNx[i] = wt[i]/pow(W,2)*(W*dNx[i]-N[i]*dWx);
    dNy[i] = wt[i]/pow(W,2)*(W*dNy[i]-N[i]*dWy);
	
	N[i] = N[i]*wt[i]/W;
    }
    
    return 0;
}


/* Main */
double **shapeFunc2(	int n, //degree
						double *uuu, //parametric t
						double *wt) //weights
{
    /*Output data*/
    int num = (n+1)*(n+2)/2;
    double *N = malloc(num*sizeof(double));
	double *dNx = malloc(num*sizeof(double));
	double *dNy = malloc(num*sizeof(double));
	double *ddNx = malloc(num*sizeof(double));
	double *ddNy = malloc(num*sizeof(double));
	double *ddNxy = malloc(num*sizeof(double));

    BernsteinBasis0(n, uuu, wt, N, dNx, dNy, ddNx, ddNy, ddNxy);
	double **result = {&N, &dNx, &dNy, &ddNx, &ddNy, &ddNxy};
	return result;
   
}
