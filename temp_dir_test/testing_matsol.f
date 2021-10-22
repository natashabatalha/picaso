      PROGRAM TEST_OF_LUDCMP
C     TESTING DONE-- LOCATE MODULE JUST FINDS THE NEAREST LOWER BOUND FROM
C     A TABLE OF THE SUPPLIED NUMBER 
C      implicit double precision (a-h,o-z)
      implicit none
      EXTERNAL LUDCMP, LUBKSB
      INTEGER N, NP, I, J, INDX(3)
      REAL*8 A(5,5) , B(3)
      REAL*8 D
      
      N=3
      NP=5
      D=1.0
      DO 1 I=1,NP
        DO 2 J=1,NP
          A(I,J)=I+2*J
  2     CONTINUE
  1   CONTINUE
      
      DO 3 I=1,N
        INDX(I)=0
        B(I)=1.0*I

  3   CONTINUE
      
      WRITE(*,*) "This is the input A array"
      WRITE(*,*) A  
      WRITE(*,*) "NOW THE POST CALL A ARRAY"
      
      
      CALL LUDCMP(A, N, NP, INDX,D)
      CALL LUBKSB(A,N,NP,INDX,B)

      WRITE(*,*) A
      WRITE(*,*) "NOW THE INDEX ARRAY"
      WRITE(*,*) INDX

      WRITE(*,*) "NOW B ARRAY"
      WRITE(*,*) B

      

      
     
      

      
      
      

      END PROGRAM TEST_OF_LUDCMP

      SUBROUTINE ludcmp(a,n,np,indx,d)
	    implicit double precision (a-h,o-z)
      INTEGER n,np,indx(n),NMAX
      dimension a(np,np)
      PARAMETER (NMAX=100,TINY=1.0d-20)
      INTEGER i,imax,j,k
      dimension vv(NMAX)
      d=1.
      do 12 i=1,n
        aamax=0.
        do 11 j=1,n
          if (abs(a(i,j)).gt.aamax) aamax=abs(a(i,j))
11      continue
        if (aamax.eq.0.) then
		   print *,'singular matrix in ludcmp'
		   stop
        endif
        vv(i)=1./aamax
12    continue
      do 19 j=1,n
        do 14 i=1,j-1
          sum=a(i,j)
          do 13 k=1,i-1
            sum=sum-a(i,k)*a(k,j)
13        continue
          a(i,j)=sum
14      continue
        aamax=0.
        do 16 i=j,n
          sum=a(i,j)
          do 15 k=1,j-1
            sum=sum-a(i,k)*a(k,j)
15        continue
          a(i,j)=sum
          dum=vv(i)*abs(sum)
          if (dum.ge.aamax) then
            imax=i
            aamax=dum
          endif
16      continue
        if (j.ne.imax)then
          do 17 k=1,n
            dum=a(imax,k)
            a(imax,k)=a(j,k)
            a(j,k)=dum
17        continue
          d=-d
          vv(imax)=vv(j)
        endif
        indx(j)=imax
        if(a(j,j).eq.0.)a(j,j)=TINY
        if(j.ne.n)then
          dum=1./a(j,j)
          do 18 i=j+1,n
            a(i,j)=a(i,j)*dum
18        continue
        endif
19    continue
      return
      END

      SUBROUTINE lubksb(a,n,np,indx,b)
	    implicit double precision (a-h,o-z)
      INTEGER n,np,indx(n)
      dimension a(np,np),b(n)
      INTEGER i,ii,j,ll
      ii=0
      do 12 i=1,n
        ll=indx(i)
        sum=b(ll)
        
        b(ll)=b(i)
        
        if (ii.ne.0)then
          do 11 j=ii,i-1
            sum=sum-a(i,j)*b(j)
11        continue
          
        else if (sum.ne.0.) then
          ii=i
        endif
        b(i)=sum
12    continue
      
      do 14 i=n,1,-1
        sum=b(i)
        do 13 j=i+1,n
          sum=sum-a(i,j)*b(j)
13      continue
        b(i)=sum/a(i,i)
14    continue
      return
      END

