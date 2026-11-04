% nmr_params.m - parameters for nmr_marginal_osc.slx
% Runs automatically as the model's InitFcn. Choose the operating mode by
% setting  nmr_mode  in the base workspace before pressing Run:
%   nmr_mode = 'scope'      100 Hz, 50 uT modulation, on resonance (default)
%   nmr_mode = 'scope_off'  as 'scope' but B0 offset +20 uT
%   nmr_mode = 'lockin'     200 Hz, 1.5 uT modulation, slow +/-30 uT B0 sweep

if ~exist('nmr_mode','var'), nmr_mode = 'scope'; end

% Spins and magnet
P.gamma = 2*pi*42.577e6;          % proton, rad/s/T
P.f0    = 20e6;                   % oscillator frequency, Hz
P.w0    = 2*pi*P.f0;
P.Bres  = P.w0/P.gamma;           % resonant field, T
P.T1    = 2e-3;  P.T2 = 1e-3;     % s, CuSO4-doped water
P.chi0  = 4e-9;                   % proton static susceptibility, SI
P.sigB  = 5e-6;                   % magnet inhomogeneity (Gaussian sigma), T
P.Niso  = 101;                    % isochromats
P.dB    = linspace(-4,4,P.Niso)'*P.sigB;
P.wiso  = exp(-0.5*(P.dB/P.sigB).^2); P.wiso = P.wiso/sum(P.wiso);

% Marginal oscillator (averaged amplitude model)
P.eta   = 0.6;                    % filling factor
P.Q0    = 80;                     % tank Q
P.eps   = 0.05;                   % marginality
P.A0    = 0.5;                    % no-signal RF amplitude at detector, V
P.As    = P.A0/sqrt(P.eps);
P.kB1   = 1e-6/P.A0;              % rotating B1 per volt
P.y0    = [zeros(2*P.Niso,1); ones(P.Niso,1); P.A0];   % [Mx; My; Mz; A]

% Modulation coils and sense resistor
P.Ncoil = 20; P.Rcoil = 0.025;
P.kcoil = 0.715*4e-7*pi*P.Ncoil/P.Rcoil;   % T/A
P.Rsense = 1;

% Detector and preamp
P.Gpre  = 20;  P.flp = 1e3;  P.fhp = 2;
P.noise = 2e-3;                   % V rms within preamp bandwidth
P.Tsn   = 1e-5;                   % noise sample time
enbw    = pi*P.flp/4;
P.noisePower = (P.noise*sqrt((1/P.Tsn)/(2*enbw)))^2*P.Tsn;

switch nmr_mode
    case 'scope'
        P.fm = 100; P.Bmod = 50e-6; P.Boff = 0;     P.sweep = 0; P.Tstop = 40e-3; P.maxStep = 2e-6;
    case 'scope_off'
        P.fm = 100; P.Bmod = 50e-6; P.Boff = 20e-6; P.sweep = 0; P.Tstop = 40e-3; P.maxStep = 2e-6;
    case 'lockin'
        P.fm = 200; P.Bmod = 1.5e-6; P.Boff = -30e-6; P.Tstop = 1.5;
        P.sweep = 60e-6/P.Tstop; P.maxStep = 1e-5;
    otherwise
        error('nmr_mode must be scope, scope_off or lockin');
end

% Lock-in
P.tau = 20e-3;                    % per-pole time constant
P.phi = 2*atan(P.fm/P.flp);       % preamp phase lag, compensated in reference
