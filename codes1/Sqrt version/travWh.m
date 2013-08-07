function out = travWh(c,uini,N,tolerance);
% travWh(speed, initial guess, N, tolerance)


% ********************************************************************* %
%
% -cu + + u^2 + K_h_0 u = 0   on 0 < x < pi
%
%
% [ -cI + Tau ] u_N  + u_N^2   =  f_N(u_N)  =  0  
%
%
% solved here via Newton iteration:
%
% DF(u_N)h = - f(u_N)  ,   u_N^+ = u_N + h
%
%
%
%  -- Here we try to find the left half of an even periodic solution --
% 
%
% ********************************************************************* %



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Basic parameters
%
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = pi;
h = L/N;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  space variable x and Fourier variable xi = k
%
%  coefficient w
%
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = (h/2:h:L-h/2)';
xi = (0:1:N-1)';
ww = sqrt(2/N)*ones(N,1);
ww(1) = sqrt(1/N);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Compute differentiation matrix D_N^2 
%
%  and operator L
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Tau=zeros(N);
for m=1:N;
   for n=1:N;
      Tau(m,n) = ww(1)*ww(1)*cos(x(n)*xi(1))*cos(x(m)*xi(1));
      for k=2:N;
      Tau(m,n) = Tau(m,n) + sqrt((1/xi(k))*tanh(xi(k)))*ww(k)*ww(k)*cos(x(n)*xi(k))*cos(x(m)*xi(k));
      end;
   end;
end;

ScriptL = -c*eye(N) + Tau;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Newton loop
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u=uini;  
change = 2; it = 0;
while change > tolerance;
%     if it > 5000;
%         break;
%     end;
     DFu = ScriptL + diag(3*(u+1).^(1/2)-3);          %square root test
     corr = -DFu\( ScriptL*u + 2*(u+1).^(3/2)-3*u-2);   %square root test
 
     
     %   DFu = ScriptL + 2*diag(u);                       %square test   
%   corr = -DFu\( ScriptL*u + u.^2);                 %square test  
%     DFu = ScriptL + 3*diag(u.^2);                  %cubic test
%     corr = -DFu\( ScriptL*u + u.^3);               %cubic test
   

    unew = u + corr;
    change = norm(corr,inf);
    u = unew; it = it+1;
    
 end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Output
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = u;

