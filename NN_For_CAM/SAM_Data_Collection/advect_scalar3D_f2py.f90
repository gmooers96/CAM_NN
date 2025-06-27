
!subroutine advect_scalar3D (f, u, v, w, rho, rhow, flux, dosubtr)
 	
subroutine advect_scalar3D_f2py (f_bound, u_bound, v_bound, w_bound, rho, rhow, dx, dy, dz, dtn,dimx,&
                dimy,dimz,adz, ady, mu, muv,  flux_x_out, flux_y_out,flux_z_out ,tend_out, dosubtr,my_val,val_min)
!     positively definite monotonic advection with non-oscillatory option

!use grid
!use terrain, only: terra ! JY: Not sure if necessary. I would need to rescale everything with terrain using a different function?
!use vars, only: misc
implicit none

integer, parameter :: ext_dim = 1
integer, parameter ::  ext_dim2 = 2
integer, parameter :: ext_dim3 = 4

real :: dx, dy, dz, dtn, dtdx, dtdy, dtdz, rhox, rhoy, rhoz
integer :: dimx,dimy,dimz

real f(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)
real f_old(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz) ! To accurately calculate the tendency

real u(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz)
real v(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz)
real w(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz + 1) !NOTE THAT I Do not have w at the upper boundary - unlike during the run

real flux_x(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)
real flux_y(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)
real flux_z(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)
real tend(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)
!real tend2(-1:dimx+ext_dim2 ,-1:dimy+ext_dim2,dimz)


real f_bound(dimx+4 ,dimy+4 ,dimz)
real u_bound(dimx+4 ,dimy+4 ,dimz)
real v_bound(dimx+4 ,dimy+4 ,dimz)
real w_bound(dimx+4 ,dimy+4 ,dimz+1) !NOTE THAT I Do not have w at the upper boundary - unlike during the run

real mu(-1:dimy+ext_dim2) ! JY How to set my, ady? They have 1 more grid point than what we have. mu(j) = cos(phi), cos(deg2rad*lat(j+jt))
real muv(-1:dimy+ext_dim2)
real ady(-1:dimy + ext_dim2)
real adz(dimz)
real rho(dimz)
real rhow(dimz + 1)
!real flux(dimz + 1)

logical dosubtr ! flag to subtract minimum before advection (recommended for temperature)
logical my_val

real val_min

real mx (0:dimx+ext_dim,0:dimy+ext_dim,dimz)
real mn (0:dimx+ext_dim,0:dimy+ext_dim,dimz)
real uuu(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz)
real vvv(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz)
real www(-1:dimx+ext_dim2,-1:dimy+ext_dim2,dimz + 1) !NOTE THAT I Do not have w at the upper boundary - unlike during the run

integer :: nzm, nz
real eps
!real iadz(dimz),irho(dimz),irhow(dimz)
real g(-1:dimy+ext_dim2,dimz),ig(-1:dimy+ext_dim2,dimz)
integer i,j,k,ic,ib,jc,jb,kc,kb
logical nonos

!real, parameter :: eps = 1.e-9
!real g(-1:nyp2,nzm),ig(-1:nyp2,nzm)
!integer i,j,k,ic,ib,jc,jb,kc,kb
!logical nonos


real x1, x2, a, b, a1, a2, y
real andiff,across,pp,pn
real fmin

real flux_x_out(dimx ,dimy,dimz)
real flux_y_out(dimx ,dimy,dimz)
real flux_z_out(dimx ,dimy,dimz)
real tend_out(dimx ,dimy,dimz)

!I am not sure if I should also get the modified f values. 
!f2py intent(inplace) flux_z_out
!f2py intent(inplace) flux_y_out
!f2py intent(inplace) flux_x_out
!f2py intent(inplace) tend_out

andiff(x1,x2,a,b)=(abs(a)-2.*a*a/b)*0.5*(x2-x1)
across(x1,a,a1,a2)=0.0625*x1*a1*a2/a
pp(y)= max(0.,y)
pn(y)=-min(0.,y)

nonos = .true.
eps = 1.e-10
nzm = dimz
nz = dimz +1
www(:,:,dimz + 1)=0.


!nonos = .true.
!www(:,:,nz)=0.

dtdx = dtn/dx
dtdy = dtn/dy
dtdz = dtn/dz

 do k=1,nzm ! Following advect_all_scalars.f90 (advect_all_scalars.f90 run before main).
  do j=-1,dimy+ext_dim2 
   !rhox = rho(k)*dtdx
   rhox = rho(k)*dtdx*adz(k)*ady(j)
   !rhoy = rho(k)*dtdy
   rhoy = rho(k)*dtdy*muv(j)*adz(k)
   !muv(j) = (ady(j-1)*mu(j)+ady(j)*mu(j-1))/(ady(j-1)+ady(j))
   !rhoz = rhow(k)*dtdz
   rhoz = rhow(k)*dtdz*ady(j)*mu(j)
   do i=-1,dimx+ext_dim2
    f(i,j,k)=f_bound(i+ext_dim2,j+ext_dim2,k)
    u(i,j,k)=u_bound(i+ext_dim2,j+ext_dim2,k) * rhox
    v(i,j,k)=v_bound(i+ext_dim2,j+ext_dim2,k) * rhoy
    w(i,j,k)=w_bound(i+ext_dim2,j+ext_dim2,k) * rhoz
   end do
  end do
 end do


!do j=-1,nyp2
do j =-1, dimy+ext_dim2 
 do k=1,nzm
  g(j,k) = rho(k)*ady(j)*adz(k)*mu(j)
  ig(j,k) = 1./g(j,k)
 end do
end do

! ------
! MK: 12/16 remove minimum for accuracy of single precision, especially for temperature

if(dosubtr) then
  if(my_val) then
    fmin = val_min
    f=f-fmin
  else
    fmin=minval(f)
    f=f-fmin
  end if
else
  fmin=0.
end if


!-----------------------------------------
	 	 
if(nonos) then

 do k=1,nzm
  kc=min(nzm,k+1)
  kb=max(1,k-1)
  do j=0,dimy+ext_dim
   jb=j-1
   jc=j+1
   do i=0,dimx+ext_dim
    ib=i-1
    ic=i+1
    mx(i,j,k)=max(f(ib,j,k),f(ic,j,k),f(i,j,kb),f(i,j,kc),f(i,j,k),f(i,jb,k),f(i,jc,k))
    mn(i,j,k)=min(f(ib,j,k),f(ic,j,k),f(i,j,kb),f(i,j,kc),f(i,j,k),f(i,jb,k),f(i,jc,k))
   end do
  end do
 end do
	 
end if  ! nonos

 do k=1,nzm
  do j=-1,dimy+ext_dim2 
   do i=0,dimx+ext_dim2 !JY change index
    ! Previously I had to limit the i-1 index to -1. Check if this is necessary here. 
    uuu(i,j,k)=max(0.,u(i,j,k))*f(i-1,j,k)+min(0.,u(i,j,k))*f(i,j,k)
   end do
  end do
 end do
 do k=1,nzm
  do j=0,dimy+ext_dim2 !JY change index
   do i=-1,dimx+ext_dim2 
        ! Previously I had to limit the i-1 index to -1. Check if this is necessary here. 
    vvv(i,j,k)=max(0.,v(i,j,k))*f(i,j-1,k)+min(0.,v(i,j,k))*f(i,j,k)
   end do
  end do
 end do
 do k=1,nzm
  kb=max(1,k-1)
  do j=-1,dimy+ext_dim2
   do i=-1,dimx+ext_dim2
    www(i,j,k)=max(0.,w(i,j,k))*f(i,j,kb)+min(0.,w(i,j,k))*f(i,j,k)
   end do
  end do
!  flux(k) = 0.
!  do j=1,ny
!   do i=1,nx
!    flux(k) = flux(k) + www(i,j,k)
!   end do
!  end do
 end do
! misc(1:nx,1:ny,1:nzm) = www(1:nx,1:ny,1:nzm)


 do k=1,nzm
  do j=-1,dimy+ext_dim2-1 !JY changed undex
   do i=-1,dimx+ext_dim2-1 !JY changed undex
      f_old(i,j,k) = f(i,j,k)
      f(i,j,k)=max(0.,f(i,j,k) &  !The max term was not in the previous version
              -(uuu(i+1,j,k)-uuu(i,j,k)+vvv(i,j+1,k)-vvv(i,j,k)+ &
                www(i,j,k+1)-www(i,j,k))*ig(j,k)) ! In previous version there was iadz multiplying only the www part.
!   tend(i,j,k)= &  ! I removed the max term here... 
!             -(uuu(i+1,j,k)-uuu(i,j,k)+vvv(i,j+1,k)-vvv(i,j,k)+ &
!               www(i,j,k+1)-www(i,j,k))*ig(j,k) ! In previous version there was iadz multiplying only the www part.

      flux_x(i,j,k) = uuu(i,j,k)
      flux_y(i,j,k) = vvv(i,j,k)
      flux_z(i,j,k) = www(i,j,k)
!      f(i,j,k)= max(0.,f(i,j,k) + tend(i,j,k))
      tend(i,j,k) = f(i,j,k) - f_old(i,j,k)
        ! This is strange   g(j,k) = rho(k)*ady(j)*adz(k)*mu(j)
        !  ig(j,k) = 1./g(j,k)  so I am not sure where the ady and adz part cancel again, also I am confused
        ! These changes happend in: u1(i,j,k) = a1*u(i,j,k)*rhox+a2*u1(i,j,k)*dtn advect_all_scalars.f90 so the velocities where
        ! multiplied by other terms... 
   end do
  end do
 end do 

 do k=1,nzm
  kc=min(nzm,k+1)
  kb=max(1,k-1)
  do j=0,dimy+ext_dim
   jb=j-1
   jc=j+1
   do i=0,dimx+ext_dim2
    ib=i-1
    uuu(i,j,k)=andiff(f(ib,j,k),f(i,j,k),u(i,j,k),g(j,k)+g(j,k)) &
              -across(f(ib,jc,k)+f(i,jc,k)-f(ib,jb,k)-f(i,jb,k), g(j,k)+g(j,k), & 
                      u(i,j,k), v(ib,j,k)+v(ib,jc,k)+v(i,jc,k)+v(i,j,k)) &
              -across(f(ib,j,kc)+f(i,j,kc)-f(ib,j,kb)-f(i,j,kb), g(j,k)+g(j,k), & 
                      u(i,j,k), w(ib,j,k)+w(ib,j,k+1)+w(i,j,k)+w(i,j,k+1))
   end do
  end do
 end do


 do k=1,nzm
  kc=min(nzm,k+1)
  kb=max(1,k-1)
  do j=0,dimy+ext_dim2
   jb=j-1
   do i=0,dimx+ext_dim
    ib=i-1
    ic=i+1
    vvv(i,j,k)=andiff(f(i,jb,k),f(i,j,k),v(i,j,k),g(j,k)+g(jb,k)) &
              -across(f(ic,jb,k)+f(ic,j,k)-f(ib,jb,k)-f(ib,j,k), g(j,k)+g(jb,k), &
                      v(i,j,k), u(i,jb,k)+u(i,j,k)+u(ic,j,k)+u(ic,jb,k)) &
              -across(f(i,jb,kc)+f(i,j,kc)-f(i,jb,kb)-f(i,j,kb), g(j,k)+g(jb,k), & 
                      v(i,j,k), w(i,jb,k)+w(i,j,k)+w(i,j,k+1)+w(i,jb,k+1))
   end do
  end do
 end do

 do k=1,nzm
  kb=max(1,k-1)
  do j=0,dimy+ext_dim
   jb=j-1
   jc=j+1
   do i=0,dimx+ext_dim
    ib=i-1
    ic=i+1
    www(i,j,k)=andiff(f(i,j,kb),f(i,j,k),w(i,j,k),g(j,k)+g(j,kb)) &
             -across(f(ic,j,kb)+f(ic,j,k)-f(ib,j,kb)-f(ib,j,k), g(j,k)+g(j,kb), &
                     w(i,j,k), u(i,j,kb)+u(i,j,k)+u(ic,j,k)+u(ic,j,kb)) &
             -across(f(i,jc,k)+f(i,jc,kb)-f(i,jb,k)-f(i,jb,kb), g(j,k)+g(j,kb), &
                     w(i,j,k), v(i,j,kb)+v(i,jc,kb)+v(i,jc,k)+v(i,j,k))
   end do
  end do
 end do

www(:,:,1) = 0.

!---------- non-osscilatory option ---------------

if(nonos) then

 do k=1,nzm
  kc=min(nzm,k+1)
  do j=0,dimy+ext_dim
   jc=j+1
   do i=0,dimx+ext_dim
    ic=i+1
     mx(i,j,k)=(mx(i,j,k)-f(i,j,k))*g(j,k)/ &
                       (pn(uuu(ic,j,k)) + pp(uuu(i,j,k))+ &
                       (pn(vvv(i,jc,k)) + pp(vvv(i,j,k)))+ &
                       (pn(www(i,j,kc)) + pp(www(i,j,k)))+eps)
     mn(i,j,k)=(f(i,j,k)-mn(i,j,k))*g(j,k)/ &
                       (pp(uuu(ic,j,k)) + pn(uuu(i,j,k))+ &
                       (pp(vvv(i,jc,k)) + pn(vvv(i,j,k)))+ &
                       (pp(www(i,j,kc)) + pn(www(i,j,k)))+eps)	
   end do
  end do
 end do


 do k=1,nzm
  do j=1,dimy
   do i=1,dimx+ext_dim
    ib=i-1
    uuu(i,j,k)=pp(uuu(i,j,k))*min(1.,mx(i,j,k), mn(ib,j,k)) &
             - pn(uuu(i,j,k))*min(1.,mx(ib,j,k),mn(i,j,k))
   end do
  end do
 end do

 do k=1,nzm
  do j=1,dimy+ext_dim
   jb=j-1
   do i=1,dimx
    vvv(i,j,k)=pp(vvv(i,j,k))*min(1.,mx(i,j,k), mn(i,jb,k)) &
             - pn(vvv(i,j,k))*min(1.,mx(i,jb,k),mn(i,j,k))
   end do
  end do
 end do

 do k=1,nzm
  kb=max(1,k-1)
  do j=1,dimy
   do i=1,dimx
    www(i,j,k)=pp(www(i,j,k))*min(1.,mx(i,j,k), mn(i,j,kb)) &
             - pn(www(i,j,k))*min(1.,mx(i,j,kb),mn(i,j,k))
!    flux(k) = flux(k) + www(i,j,k)
   end do
  end do
 end do

! misc(1:nx,1:ny,1:nzm) = misc(1:nx,1:ny,1:nzm)+www(1:nx,1:ny,1:nzm)

endif ! nonos

!if(collect_coars) then
!  dtdz = dtn/dz
!  do k=1,nzm
!    do j=1,ny
!     rhoz = 1./(dtdz*ady(j)*mu(j)) NOTE THAT THERE IS A DIFFERENT VERSION FOR THIS LINE
!     do i=1,nx
!       misc(i,j,k) = misc(i,j,k)*rhoz
!!     end do
!    end do
!  end do
!end if
do k=1,nzm
 kc=k+1
 do j=1,dimy
 !rhox = 1./(rho(k)*dtdx*adz(k)*ady(j)) ! JY wasnt sure if we need the density here. 
   !rhoy = 1./(rho(k)*dtdy*muv(j)*adz(k))
   !rhoz = 1./(rhow(k)*dtdz*ady(j)*mu(j))

   rhox = 1./(dtdx*adz(k)*ady(j))
   rhoy = 1./(dtdy*muv(j)*adz(k))
   rhoz = 1./(dtdz*ady(j)*mu(j))
  do i=1,dimx
 ! MK: added fix for very small negative values (relative to positive values) 
 !     especially  when such large numbers as
 !     hydrometeor concentrations are advected. The reason for negative values is
 !     most likely truncation error.
   f_old(i,j,k) = f(i,j,k)+fmin
   f(i,j,k)=max(0.,f(i,j,k)+fmin &
                  -(uuu(i+1,j,k)-uuu(i,j,k)+vvv(i,j+1,k)-vvv(i,j,k)+ &
                    www(i,j,k+1)-www(i,j,k))*ig(j,k)) !*terra(i,j,k))


   tend(i,j,k) = tend(i,j,k)  + f(i,j,k) - f_old(i,j,k)
   flux_x(i,j,k) = (flux_x(i,j,k) + uuu(i,j,k)) * rhox
   flux_y(i,j,k) = (flux_y(i,j,k) + vvv(i,j,k)) * rhoy
   flux_z(i,j,k) = (flux_z(i,j,k) + www(i,j,k)) * rhoz
   !flux_z(i,j,k) = (flux_z(i,j,k)) * rhoz
   flux_x_out(i,j,k) = flux_x(i,j,k)
   flux_y_out(i,j,k) = flux_y(i,j,k)
   flux_z_out(i,j,k) = flux_z(i,j,k)
   tend_out(i,j,k) = tend(i,j,k)

  end do
 end do
end do 

end subroutine advect_scalar3D_f2py


