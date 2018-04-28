%% Filter Design
%% FIR and IIR Filter Design
%------------------------------------------------------------------------
% FIR and IIR Filter Design using Filter Design Tools
% Written by Abhiram S
%------------------------------------------------------------------------
%% Input Signal x(n)
% Normalized frequency components
f = [0.1 0.2 0.5 0.65 0.75 1.1 1.3 1.75 1.9];
N = 400;  % No. of samples
n = 1:N;
x = 0;
 
% Signal Generation
for i = 1:length(f)
    x = x + exp(1i*pi*f(i)*n);
end

% Frequency Spectrum Plot of x(n)
Xk = fft(x);
plot(n*2/max(n),abs(Xk));
xlabel('Normalized Frequency (\times\pi rad)');
ylabel('|X(\omega)|');
title('Frequency Spectrum of x(n)');

%% Background Noise Addition
bn = length(f)/20*(randn(1,N)+1i*randn(1,N));
x = x + bn;

% Frequency Spectrum Plot of x(n) with Background Noise
Xk = fft(x);
plot(n*2/max(n),abs(Xk));
xlabel('Normalized Frequency (\times\pi rad)');
ylabel('|X_{n}(\omega)|');
title('Frequency Spectrum of x_{n}(n)');

%% Designing Low Pass FIR Blackman Window Filter
% Frequency Normalized to 1
Nr   = 110;      % Order
Fc   = 0.4;      % Cutoff Frequency
flag = 'scale';  % Sampling Flag
 
% Create the window vector for the design algorithm
win = blackman(Nr+1);
 
% Calculate the coefficients using the FIR function
b = fir1(Nr, Fc, 'low', win, flag);
Hlp = dfilt.dffir(b);

%% Plot the magnitude response of FIR filter
fvtool(Hlp,'analysis','magnitude');

%% Plot the phase response of FIR filter
fvtool(Hlp,'analysis','phase');

%% Plot the group delay of FIR filter
fvtool(Hlp,'analysis','grpdelay');

%% Calculate FIR Filtered Output
y = filter(Hlp,x);

% Plot Frequency Spectrum of Output
Yk = fft(y);
plot(n*2/max(n),abs(Yk));
xlabel('Normalized Frequency (\times\pi rad)');
ylabel('|Y(\omega)|');
title(['Low Pass Filtered Output (F_{c}=0.4) ',...
'using Blackman Window FIR Filter']);

%% Designing Band Pass IIR Butterworth Filter
% Frequency Normalized to 1
Fstop1 = 0.35;  % First Stopband Frequency
Fpass1 = 0.45;  % First Passband Frequency
Fpass2 = 0.75;  % Second Passband Frequency
Fstop2 = 0.85;  % Second Stopband Frequency
Astop1 = 80;    % First Stopband Attenuation (dB)
Apass  = 1;     % Passband Ripple (dB)
Astop2 = 80;    % Second Stopband Attenuation (dB)
 
h = fdesign.bandpass('fst1,fp1,fp2,fst2,ast1,ap,ast2', ...
    Fstop1, Fpass1, Fpass2, Fstop2, Astop1, Apass, Astop2);
 
Hbp = design(h, 'butter', 'MatchExactly', ...
      'stopband', 'SOSScaleNorm', 'Linf');
  
%% Plot the magnitude response of IIR filter
fvtool(Hbp,'analysis','magnitude');

%% Plot the phase response of IIR filter
fvtool(Hbp,'analysis','phase');

%% Plot the group delay of IIR filter
fvtool(Hbp,'analysis','grpdelay');

%% Calculate IIR Filtered Output
y = filter(Hbp,x);

% Plot Frequency Spectrum of Output
Yk = fft(y);
plot(n*2/max(n),abs(Yk));
xlabel('Normalized Frequency (\times\pi rad)');
ylabel('|Y(\omega)|');
title(['Band Pass Filtered Output (F_{c1}=0.4,F_{c2}=0.8) ',...
'using IIR Butterworth Filter']);