tic;

branWh;                        % Compute branch of traveling waves



NN = 2*N;                       % This is the new N for the FFT on [0,2pi]
hh = 2*L/NN;
xx = (0:hh:2*L-hh);  
uu = interpol(u,L,N,NN);        % interpolate to FFT grid


testWh(c,uu, NN, 0.001, 1)      % c is wavespeed, uu is periodic wave profile on [0,2pi]
toc;                                % NN is number of modes
                                % 0.001 is time step
                                % 1 is number of temporal periods

