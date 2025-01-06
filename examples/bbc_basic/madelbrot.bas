      REM Target BBC Micro
      MODE 14
      xmin=-2.5:xmax=1
      xwidth=xmax-xmin
      ymin=-1:ymax=1
      ywidth=ymax-ymin
      xsize%=1280:ysize%=1024:REM this is the physical size of the MODE 2 screen
      :
      max%=1024:REM maximum iterations
      :
      PROCplot(0,0)
      PROCplot(2,2)
      PROCplot(0,2)
      PROCplot(2,0)
      PROCplot(1,1)
      PROCplot(3,3)
      PROCplot(1,3)
      PROCplot(3,1)
      PROCplot(1,0)
      PROCplot(3,2)
      PROCplot(1,2)
      PROCplot(3,0)
      PROCplot(0,1)
      PROCplot(2,3)
      PROCplot(0,3)
      PROCplot(2,1)
      :
      REPEAT UNTIL INKEY(0)<>-1
      END
      :
      DEFPROCplot(f%,g%)
      FOR X%=0 TO (xsize%-1) STEP 4
        FOR Y%=0 TO (ysize%-1) STEP 4
          a=(xwidth*(X%+f%)/xsize%)+xmin
          b=(ywidth*(Y%+g%)/ysize%)+ymin
          PROCit(a,b,max%)
          h%=7-7*LOG(IT%)/LOG(max%)
          IF (ABS(e)+ABS(f))>4 GCOL0,h% ELSE GCOL 0,0
          MOVE (X%+f%)*2,(Y%+g%)*2:DRAW (X%+f%)*2,(Y%+g%)*2
        NEXT Y%
      NEXT X%
      ENDPROC
      :
      DEFPROCit(a,b,ITER%)
      IT%=0
      e=0
      f=0
      REPEAT
        u=(e*e)-(f*f)+a
        v=(2*e*f)+b
        e=u
        f=v
        IT%=IT%+1
      UNTIL IT%=ITER% OR (ABS(e)+ABS(f))>4
      ENDPROC
