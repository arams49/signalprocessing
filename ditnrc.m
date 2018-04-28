function Xf = ditnrc(x,M)
%------------------------------------------------------------------------ 
% Function to compute Fast Fourier Transform (FFT)
%  using Decimation in Time (DIT) method
% DIT is implemented in a Non Recursive fashion
% Written by Abhiram S
%------------------------------------------------------------------------
% Input Arguments = (x,M)
% x = Input Signal
% M = Length of the input signal
%------------------------------------------------------------------------
% Output Arguments = [Xf]
% Xf = Discrete Fourier Transform of input signal
%------------------------------------------------------------------------

x=x(1:M);  % Input Signal
S=ceil(log2(M));  % Number of stages in DIT
N=2^S;  % New length of input sequence and its DFT
Xf=[x, zeros(1,N-M)];
Xf=bitrevorder(Xf);  % Input in Bit Reversed Order

% Non Recursive DIT
for s=1:S
    B=2^s;
    k=0:B/2-1;
    W=exp(-1i*2*pi*k/B);
    for r=1:B:N
        b1=r+B/2-1;
        b2=r+B-1;
        x1=Xf(r:b1);
        x2=W.*Xf(b1+1:b2);
        Xf(r:b2)=[x1+x2,x1-x2];
    end
end
end