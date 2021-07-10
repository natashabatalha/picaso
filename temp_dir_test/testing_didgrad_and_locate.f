      PROGRAM TEST_OF_DIDGRAD
C     TESTING DONE-- LOCATE MODULE JUST FINDS THE NEAREST LOWER BOUND FROM
C     A TABLE OF THE SUPPLIED NUMBER 
C      implicit double precision (a-h,o-z)
      implicit none
      EXTERNAL LOCATE, DIDGRAD
      INTEGER N, I, J
      REAL*8 P_TABLE(26), T_TABLE(53), T_SEARCH, P_SEARCH
      REAL*8 GRAD(53,26), GRADX, Ttest, Ptest

      
      
      
      
      DO 1 I=1,53
        
        DO 2 J=1,26
            GRAD(I,J)=I+J-2.0
  2     CONTINUE
  1   CONTINUE
      
      DO 3 I=1,53
        T_TABLE(I)=I-1
  3   CONTINUE
      
      DO 4 I=1,26
        P_TABLE(I)=I-1
  4   CONTINUE
      Ttest= 200
      Ptest= 303
      CALL didgrad(Ttest, Ptest,GRADX, T_table, P_TABLE, GRAD)

      

      WRITE(*,*) GRADX
     
      

      
      
      

      END PROGRAM TEST_OF_DIDGRAD

        subroutine didgrad(t,p,gradx,tlog,plog,grad)
            implicit double precision (a-h,o-z)
c            common /cps/ tlog(53),plog(26),cp(53,26),grad(53,26)
            REAL*8 tlog(53),plog(26),grad(53,26)
            tl = dlog10(t)
            pl = dlog10(p)
            
        CALL LOCATE(tlog,53,tl,kt)  
        CALL LOCATE(plog,26,pl,kp)  

        ipflag = 0
        if (kp.eq.0) then
C       we are at low pressure, use the lowest pressure point
            factkp = 0.0
            kp = 1
        ipflag = 1
        endif

        if (kp.ge.26) then
c       we are at high pressure, use the highest pressure point
            factkp = 1.0
            kp = 25
        ipflag = 1
        endif

        itflag = 0
        if (kt.ge.53) then
c       we are at high temp, use the highest temp point
            factkt = 1.0
            kt = 52
        itflag = 1
        endif

        if (kt.eq.0) then
c       we are at temp, use the highest temp point
            factkt = 0.0
            kt = 1
        itflag = 1
        endif


        if (((kp.gt.0).and.(kp.lt.26)).and.(ipflag.eq.0)) then
c        print *,'kp',kp,plog(kp),pl,plog(kp+1)
            FACTkp= (-Plog(Kp)+Pl)/(Plog(Kp+1)-Plog(Kp))
        endif

        if (((kt.gt.0).and.(kt.lt.53)).and.(itflag.eq.0)) then
            FACTkt= (-Tlog(Kt)+Tl)/(Tlog(Kt+1)-Tlog(Kt))
        endif

        gp1 = grad(kt,kp)
        gp2 = grad(kt+1,kp)
        gp3 = grad(kt+1,kp+1)
        gp4 = grad(kt,kp+1)

        gradx = (1.d0-factkt)*(1.d0-factkp)*gp1 +
     &      factkt*(1.d0-factkp)*gp2 + factkt*factkp*gp3 + 
     &   (1.d0-factkt)*factkp*gp4
        RETURN
        END

      SUBROUTINE LOCATE(XX,N,X,J)
        implicit double precision (a-h,o-z)
C	Table searching routine from Numerical Recipes.  For
C       N=14 it is about twice as fast as the previous method.
	DIMENSION XX(N)
	
	JL=0
	JU=N+1
10	IF (JU-JL.GT.1) THEN
   	  JM=(JU+JL)/2
	  IF ((XX(N).GT.XX(1)).EQV.(X.GT.XX(JM))) THEN
	    JL=JM
	  ELSE
	    JU=JM
	  ENDIF
        GOTO 10
	ENDIF
	J=JL
	RETURN
	END
