%
% This program computes a branch of traveling waves of 
%
% -cu + + u^2 + K_h_0 u = 0   on 0 < x < pi
%

%clear;

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
N = 2*2*16; 
h = L/N;


x = (h/2:h:L-h/2)';   % cosine grid [h/2, 3h/2, 5h/2, .... , L-3h/2, L-h/2]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Initial guess
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cstar = sqrt(tanh(1));
c = 0.99*cstar;
vini = -0.015-0.1*cos(x);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Call functions
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tolerance =  1e-12;
u = travWh(c,vini,N,tolerance);


for ind=2:40;
   c=c-0.0025;
   u = travWh(c,u,N,tolerance);
end;

plot(x,u);
