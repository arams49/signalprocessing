%% Noise Removal using RLS Adaptive Filtering
%% RLS Algorithm for Noise Removal
%------------------------------------------------------------------------
% Recursive Least Square is an adaptive filtering technique
% A variation of the adaptive filtering, where delayed noisy signal
%  is used to model the FIR filter, is implemented for noise removal
% The noisy signal is then passed through the FIR Filter
% Written by Abhiram S
% Group Members - Ashwin Natraj, Navaneeth, Prakhar, Abhiram
%------------------------------------------------------------------------
%% Block Diagram
clear,clc,close all
imshow('RLSNoiseFilter.png');


%% Inputs and Parameters
M = 600;  % Data Points in the Input Signal
m = 1:M;
f1 = 100;  % Frequency Component 1 of Input
f2 = 350;  % Frequency Component 2 of Input
f3 = 700;  % Frequency Component 3 of Input
fs = 2000;  % Sampling Frequency of Input
SNR = -5;  % Signal to Noise Ratio
Xn = cos(2*pi*m*f1/fs) + sin(2*pi*m*f2/fs) + sin(2*pi*m*f3/fs);  % Input
Yn = awgn(Xn,SNR,'measured');  % Input with added White Gaussian Noise


%% Delayed Noisy Signal for Noise Removal
N = floor(M/4);  % FIR Filter Length
r = N;  % Delay in Noisy Input Signal
Ynr = [zeros(1,r),Yn];  % Delayed Noisy Input
Ynr = Ynr(1:M);
Xn = Xn(:);
Yn = Yn(:);
Ynr = Ynr(:);


%% RLS Filter Initialization  
k = 1.001;  % Inverse of Forgetting Factor
h = zeros(N,1);  % FIR Filter Impulse Response
d = 100 * var(Yn);
P = d * eye(N,N);
er = [];  % Error signal


%% RLS Filter Implementation
for i = 1:1:(M-N+1)
    index = i+N-1:-1:i;
    X = Ynr(index);
    Y = Yn(i+N-1);
    g = k * P * X / (1 + k*X'*P*X);
    a = Y - X'*h;
    h = h + g*a;
    P = k*P - k*g*X'*P;
    er = [er,a];
end
Ynr = Ynr(r+1:end);
Zn = conv(Ynr,h);  % Filtered Signal


%% Signal Plot of Input and Filtered Output
figure;
subplot(3,2,1);
plot(Xn);
xlabel('n');  ylabel('x(n)');
title('Noiseless Signal x(n)');
subplot(3,2,3);
plot(Yn);
xlabel('n');  ylabel('y(n)');
title('Noisy Signal y(n)');
subplot(3,2,5);
plot(Zn);
xlabel('n');  ylabel('z(n)');
title('Recovered Signal z(n)');

k = -M/2:M/2-1;
kz = -length(Zn)/2:length(Zn)/2-1;
Xf = fftshift(fft(Xn));
Yf = fftshift(fft(Yn));
Zf = fftshift(fft(Zn));

subplot(3,2,2);
plot(k*fs/M,abs(Xf));
axis tight;
xlabel('Frequency (Hz)');  ylabel('|X(f)|');
title('Frequency Spectrum of x(n)');
subplot(3,2,4);
plot(k*fs/M,abs(Yf));
axis tight;
xlabel('Frequency (Hz)');  ylabel('|Y(f)|');
title('Frequency Spectrum of y(n)');
subplot(3,2,6);
plot(kz*fs/length(Zn),abs(Zf));
axis tight;
xlabel('Frequency (Hz)');  ylabel('|Z(f)|');
title('Frequency Spectrum of z(n)');


%% Error Plot of RLS Filter
figure;
plot(1:length(er),abs(er));
xlabel('n');
ylabel('Error |e(n)|');
title('Error Plot of RLS Filter');