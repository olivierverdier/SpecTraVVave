%Solution of Whitham's equation with week dispertion
%isospectral integration of u_t+uu_x+K*u=0
%K=Fourier transform of dis(k)

%input('N=')
%N=128; L=2*pi;
N=512; L=8;
a=L/(2*pi); 
T=1.1; %input('T=')
h=2*pi/N; dt=0.005;
y=(h:h:2*pi)'; 
c1=2;
%u=uu; 
u=exp(-c1*a*0.5*(y-pi).^2)/a;
k=[(1:(N/2))';(1-N/2:-1)']; %
dis=[0; sqrt(tanh(k)./k)];
dis(N/2+1)=0;
Dv=[(0:(N/2-1))'; 0; (1-N/2:-1)'];
m=a^(-3)*.5*dt*dis.^3; 
d1=(1+i*m)./(1-i*m);
d2=-0.5*i*dt*Dv./(1-i*m);
d3=0.5*d2;
sol=plot(y, a*u, 'r', 'EraseMode', 'background');
box off; axis([0 L -0.2 1.75]);
title('Whitham equation: week dispersion'); 
text(1, 1.6, 'dispersion relation =(tanh(k)/k)^{1/2}')
text(1, 1.4, 'u_t+uu_x+K*u=0, u(x,0)=2\pi exp(-2(x-4)^2)/8');
t=0;
while t<T;
    fftu=fft(u); fftuu=fft(u.^2);
    v=real(ifft(d1.*fftu+0.5*d2.*fftuu));
    w=real(ifft(d1.*fftu+d2.*fftuu));
    w_old=w;
    eps=1e-8;
    err=1;
    while( err > eps)
        w=v+real(ifft(d3.*fft(w_old.^2)));
        err=norm(w-w_old,2);
        w_old=w;
    end
    u=w; t=t+dt; set(sol, 'ydata',a*u); pause(0.01);
    drawnow;
end