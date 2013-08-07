function out = interpol(data,L,N,NN);
% interpol(data,N,NN);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%
%
%  Interpolate to Fourier grid
% 
%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = NN;               % This is the new N
hm = 2*L/M;
xx = (0:hm:2*L-hm);  

xi = (0:1:N-1);
ww = sqrt(2/N)*ones(N,1);
ww(1) = sqrt(1/N);


Y = dct(data);




for m=1:NN;
uuu(m) =0;
for n=1:N;
uuu(m) = uuu(m) + ww(n)*Y(n)*cos(xx(m)*xi(n));
end;
end;


out=uuu';


