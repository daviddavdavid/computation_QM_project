//Calculates the ground state energy of helium.  Written by Peter Siegel
//and Shane Hall.
//To compile in linux:  gcc hegs.c -lm
//To run in linux:  ./a.out
//Our result is                 E binding energy = 77.871 eV

#include <stdio.h>
#include <math.h>


const int   imax = 15000;  //for imax=15000 and del=0.001, rmax=15 angstroms
const float  hc = 1973.285, fsc = 14.3998, mc2 = 511003.4; //Physical Constants
double chi[40002]; //wave function
double v[40002]; //Screening potential
double qtot, rho, a;
double sum;
double aa,bb,norm;
double vcheck;
double r[40002],fac,e,test0,test1,dele,del,psitest1, psitest2, c;
int i,j,l,stop,z,n, r1, r2,itest;

void potential(); //calculates the screening potential
void inite(); //determines the trial starting 1s wave function 
void ecalc(); //calculates the energy E' and wave function for the 1s electrons
void intercor(); //calculates A and B from text

void potential()
//Calculates the screening potential, V_{C1s} from text. v equals V_{C1s} from the
//text.  Gauss's law is used as described below Eq. 59 in the text
{
	qtot=0;
	v[0]=1;
	for(i=1;i<imax;i++)
	{
		qtot=qtot+chi[i]*chi[i]*del;
		v[i]=v[i-1]-qtot*fsc*del/r[i]/r[i];
	}
	vcheck=fsc/r[imax-3];
//	printf("qtot is %f   \n", qtot);
	for(i=0;i<imax;i++)
	{
		v[i]=v[i]+vcheck-v[imax-3];
	}
	return;
}

void inite()
//calculates the initial trial energy and wave function
{
	chi[0] = 0.0;
	chi[1] = 1.0;
	e=60; //initial starting energy
	dele=1; //energy step
	e=e*.99;
	for (i=1; i<imax; i++)
	{
		r[i] = i*del;
 		fac = z*fsc/r[i];//only the nuclear potential used.  No screening potential
		chi[i+1] = 2*chi[i] - chi[i-1] + chi[i]*pow(del,2)*l*(l+1)/r[i]/r[i] 
                         + pow(del,2)*chi[i]*2*mc2/hc/hc*(e-fac); //Eq. 10 in text
	}
	test0 = chi[imax-2]; //wave function at rmax
	for (j=1; j<50; j++) //iterate 50 times to "zero in" on e
	{
		e = e-dele; //change e by -dele
		for (i=1; i<imax; i++)
		{
			r[i] = i*del;
			fac = z*fsc/r[i];
			chi[i+1] = 2*chi[i] - chi[i-1] + chi[i]*pow(del,2)*l*(l+1)/r[i] 
                                 +pow(del,2)*chi[i]*2*mc2/hc/hc*(e-fac); //Eq. 10 in text
		}
		test1 = chi[imax-2]; //wave function at rmax for e-dele
		if (test1*test0 < 0) //check to see if the wave function changed sign at rmax
		{
			dele = -dele/2; //if sign change, then change direction and size of dele
		}
	test0 = test1; //update the value of wave function at rmax
	}
        printf("first e is %f \n", e);
//find value r where wave function is a minimum, for normalization
        psitest1=fabs(chi[imax-2]);
        psitest2=fabs(chi[imax-3]);
        itest=imax-4;
        while (psitest2<=psitest1)
        {
         psitest1=psitest2;
         psitest2=fabs(chi[itest]);
         itest=itest-1;
        }
//        printf("itest is %d\n",itest); //itest equals i_{match} in text
        for (i=itest; i<imax; i++)
         chi[i]=0.0;
// Normalize wave function
	sum=0;
	for(i=1;i<imax;i++)
	{
		r[i]=i*del;
		sum=sum+chi[i]*chi[i]*del;
	}
	norm=1/sqrt(sum);
        for(i=1;i<imax;i++)
          chi[i]=norm*chi[i];
	return;
}

void ecalc()
{
	dele=1; //initial change in energy e'
	e=e*.99;
	chi[0]=0;
	chi[1]=1;
	for(i=1;i<imax;i++)
	{
		r[i]=i*del;
		fac=z*fsc/r[i]-v[i]; //nucleus Coulomb + screening potential
		chi[i+1]=2*chi[i]-chi[i-1]+chi[i]*pow(del,2)*l*(l+1)/r[i]
                        +pow(del,2)*chi[i]*2*mc2/hc/hc*(e-fac);// Eq. 10 in text
	}
	test0=chi[imax-2];// value of wave function at rmax
	for(j=1;j<100;j++) //iterate 100 times to "zero in" on e
	{
		e=e-dele;
		for(i=1;i<imax;i++)
		{
			r[i]=i*del;
			fac=z*fsc/r[i]-v[i];
			chi[i+1]=2*chi[i]-chi[i-1]+chi[i]*pow(del,2)*l*(l+1)/r[i]
                                +pow(del,2)*chi[i]*2*mc2/hc/hc*(e-fac);//Eq.10 in text
		}
		test1=chi[imax-2];//value of wave function at rmax at e-dele
		if(test0*test1<0) //check to see if wave function has changed sign
		{
			dele=-dele/2; //if sign change change direction and size of dele
		}
		test0=test1; //update the value of wave function at rmax
	}
//        printf(" e is %f \n", e);
        psitest1=fabs(chi[imax-2]);
        psitest2=fabs(chi[imax-3]);
        itest=imax-4;
        while (psitest2<=psitest1) //find value of itest where the wave function is a minimum
        {
         psitest1=psitest2;
         psitest2=fabs(chi[itest]);
         itest=itest-1;
        }
//        printf("itest is %d\n",itest);
//itest here equals i_{match} in the text
        for (i=itest; i<imax; i++)
         chi[i]=0.0;
//Normalize the wave function
	sum=0;
	for(i=1;i<imax;i++)
	{
		r[i]=i*del;
		sum=sum+chi[i]*chi[i]*del;
	}
	norm=1/sqrt(sum);
        for(i=1;i<imax;i++)
          chi[i]=norm*chi[i]; //normalize the wave function
	return;
}

void intercor()
//Calculates the constants A and B, defined below Eq. 29 in the text.  A (from the text)
//equals -aa, and is the average kinetic energy.  B (from the text) equals -bb, and is
//the average Coulomb potential energy.
{
        sum=0;
        for(i=1;i<itest;i++){
                r[i]=(i)*del;
                sum=sum+chi[i]*chi[i]/r[i]*del;
        }
        bb=sum*2*fsc;
//        printf("B= %f \n", aa);
        sum=0;
        for(i=1;i<itest;i++){
                sum=sum+chi[i]*(chi[i+1]+chi[i-1]-2*chi[i]);
        }
        aa=sum*hc*hc/2/mc2/del;
//        printf("B= %f \n", aa); 
        return;
}


int main()
{
	l=0;
	z=2;
	del=0.001; //del r in angstroms
	inite(); //get an initial starting value for the 1s wave function
	  for(n=1;n<15;n++) //iterate till self-consistancy and convergence
	   {
            e=60.0; //starting energy for the 1s level
	    potential(); //calculate the Coulomb screening potential
	    ecalc(); //calculate the energy and wave function for 1s electron
            intercor(); //calculate A and B
            printf("e'= %f   A= %f    B=%f  E_binding= %f  \n",-e,-aa,-bb,e+aa+bb);
           }
         printf("\n Final Results \n \n");//print final results
         printf("e'= %f   A= %f    B=%f   \n",-e,-aa,-bb);
         e=e+aa+bb;
         printf("The 1s binding energy is %f \n",e);
	return 0;
}