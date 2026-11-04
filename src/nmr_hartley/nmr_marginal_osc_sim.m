%% NMR marginal-oscillator simulation (JFET Hartley, protons, ~20 MHz)
% Ensemble of Bloch-equation isochromats (doped water in an inhomogeneous
% magnet) coupled to an averaged (van der Pol) amplitude model of a
% marginal oscillator, followed by detector, preamp, scope and lock-in.
%
% Coupling: spins add a tank loss  1/Q = 1/Q0 + eta*chi''  where
%   chi'' = mu0*Mv/(2*B1)  (rotating-frame absorption component)
% Oscillator amplitude:
%   dA/dt = (w0/2Q0)*(eps - (A/As)^2)*A - (w0/2)*eta*chi''*A
% so with no signal A0 = As*sqrt(eps), and dA/A ~ -Q0*eta*chi''/(2*eps).
%
% Base MATLAB only. Figures are exported as PNGs next to this file.

clear; close all;
outdir = fileparts(mfilename('fullpath'));
if isempty(outdir), outdir = pwd; end

%% ---------------- Parameters ----------------
P.gamma  = 2*pi*42.577e6;   % proton gyromagnetic ratio, rad/s/T
P.f0     = 20.0e6;          % oscillator frequency, Hz
P.w0     = 2*pi*P.f0;
P.Bres   = P.w0/P.gamma;    % resonant field, T
P.T1     = 2e-3;            % s  (CuSO4-doped water)
P.T2     = 1e-3;            % s
P.chi0   = 4.0e-9;          % proton static (Curie) susceptibility of water, SI
P.eta    = 0.6;             % filling factor
P.Q0     = 80;              % tank Q without spins
P.eps    = 0.05;            % marginality (excess loop gain above threshold)
P.A0     = 0.5;             % no-signal RF amplitude at detector, V
P.As     = P.A0/sqrt(P.eps);
P.kB1    = 1e-6/P.A0;       % rotating B1 per volt (1 uT at A0)
P.sigB   = 5e-6;            % magnet inhomogeneity over sample (Gaussian sigma), T

% Modulation coils (Helmholtz pair) and current-sense resistor
P.Ncoil  = 20;              % turns per coil
P.Rcoil  = 0.025;           % coil radius, m
P.kcoil  = 0.715*4e-7*pi*P.Ncoil/P.Rcoil;   % T per A at centre
P.Rsense = 1.0;             % ohm

% Detector / preamp
P.Gpre   = 20;              % preamp gain
P.flp    = 1e3;             % preamp low-pass, Hz (two cascaded 1st-order poles)
P.noise  = 2e-3;            % detector noise, V rms within the preamp bandwidth

rng(1);
fprintf('\n=== NMR marginal oscillator simulation ===\n');
fprintf('Resonant field B0          : %.5f T  (f0 = %.3f MHz)\n', P.Bres, P.f0/1e6);
fprintf('Homogeneous linewidth      : %.0f Hz FWHM (1/pi/T2)\n', 1/(pi*P.T2));
fprintf('Inhomogeneous width        : %.0f Hz FWHM (2.355*sigma)\n', ...
        2.355*P.gamma*P.sigB/(2*pi));
s = (P.gamma*P.kB1*P.A0)^2*P.T1*P.T2;
fprintf('Saturation factor s=g^2B1^2T1T2 at A0: %.3f\n', s);
chi_pk = 0.5*P.chi0*P.w0*P.T2/(1+s);
fprintf('Peak chi'''' (homogeneous)  : %.2e\n', chi_pk);
fprintf('Small-signal dip estimate  : %.1f %% (homogeneous upper bound)\n', ...
        100*P.Q0*P.eta*chi_pk/(2*P.eps));

%% ---------------- Run 1: scope view (large modulation) ----------------
fm    = 100;                 % modulation frequency, Hz
Bmod  = 50e-6;               % modulation amplitude, T
Imod  = Bmod/P.kcoil;
fprintf('\nScope run: fm = %d Hz, Bmod = %.0f uT -> coil current %.1f mA pk\n', ...
        fm, Bmod*1e6, Imod*1e3);

dt1 = 1e-6; T1end = 40e-3; keep = 20e-3;   % show the last two modulation periods
P1  = set_isochromats(P, 201);
offsets = [0, 20e-6];
R1 = cell(1,numel(offsets));
for j = 1:numel(offsets)
    Bext = @(t) offsets(j) + Bmod*sin(2*pi*fm*t);
    tic; R1{j} = run_sim(P1, dt1, T1end, Bext); tr = toc;
    R1{j} = detector_chain(R1{j}, P, dt1);
    R1{j}.ch1 = (R1{j}.Bmod - offsets(j))/P.kcoil*P.Rsense;        % sense-resistor voltage, V
    fprintf('  offset %+5.1f uT: sim %.1f s, max dip %.2f %% of A0\n', ...
        offsets(j)*1e6, tr, 100*(1-min(R1{j}.A(end-round(keep/dt1)+1:end))/P.A0));
end

% --- Figure 1: YT display (trigger on modulation sync) ---
f1 = figure('Name','Scope YT','Color','w','Position',[100 100 900 650]);
tl = tiledlayout(f1, numel(offsets), 1, 'TileSpacing','compact');
for j = 1:numel(offsets)
    R = R1{j}; idx = (numel(R.t)-round(keep/dt1)+1):numel(R.t);
    tt = (R.t(idx)-R.t(idx(1)))*1e3;
    ax = nexttile(tl);
    yyaxis(ax,'left');  plot(ax, tt, R.ch1(idx)*1e3, 'LineWidth',1);
    ylabel(ax,'CH1 sense (mV)');
    yyaxis(ax,'right'); plot(ax, tt, R.vpre_ac(idx), 'LineWidth',1);
    ylabel(ax,'CH2 preamp (V)');
    grid(ax,'on'); xlabel(ax,'time (ms)');
    title(ax, sprintf('B_0 offset %+g \\muT  (%s)', offsets(j)*1e6, ...
        ternary(offsets(j)==0,'on resonance: blips evenly spaced','off resonance: blips bunch up')));
end
title(tl, sprintf('YT view, f_m = %d Hz, B_{mod} = %g \\muT, f_0 = %.1f MHz', fm, Bmod*1e6, P.f0/1e6));
exportgraphics(f1, fullfile(outdir,'fig1_scope_YT.png'), 'Resolution', 150);

% --- Figure 2: XY display, raw and with phase-matched X channel ---
R = R1{1}; idx = (numel(R.t)-round(keep/dt1)+1):numel(R.t);
xraw  = R.ch1;
xcomp = lp1(lp1(xraw, P.flp, dt1), P.flp, dt1);   % same lag as preamp -> traces merge
f2 = figure('Name','Scope XY','Color','w','Position',[150 150 900 400]);
tl2 = tiledlayout(f2,1,2,'TileSpacing','compact');
ax = nexttile(tl2); plot(ax, xraw(idx)*1e3, R.vpre_ac(idx), 'LineWidth',1);
grid(ax,'on'); xlabel(ax,'X = CH1 (mV)'); ylabel(ax,'Y = CH2 (V)');
title(ax,'Raw XY: preamp lag splits sweeps');
ax = nexttile(tl2); plot(ax, xcomp(idx)*1e3, R.vpre_ac(idx), 'LineWidth',1);
grid(ax,'on'); xlabel(ax,'X = CH1 via phase shifter (mV)'); ylabel(ax,'Y = CH2 (V)');
title(ax,'Phase-matched X: residual split = spin passage lag');
title(tl2,'XY view (on resonance)');
exportgraphics(f2, fullfile(outdir,'fig2_scope_XY.png'), 'Resolution', 150);

%% ---------------- Run 2: lock-in, slow field sweep, small modulation ----------------
fm2   = 200;        % Hz
bm2   = 1.5e-6;     % T, well below the linewidth -> derivative signal
Bspan = 30e-6;      % sweep +/- Bspan
Tsw   = 1.5;        % s
dt2   = 5e-6;
P2    = set_isochromats(P, 101);
Bext2 = @(t) -Bspan + 2*Bspan*t/Tsw + bm2*sin(2*pi*fm2*t);
fprintf('\nLock-in run: fm = %d Hz, Bmod = %.1f uT, sweep +/-%g uT in %.1f s ...\n', ...
        fm2, bm2*1e6, Bspan*1e6, Tsw);
tic; R2 = run_sim(P2, dt2, Tsw, Bext2); fprintf('  sim %.1f s\n', toc);
R2 = detector_chain(R2, P, dt2);

% Lock-in: reference in phase with coil current; correct for preamp phase lag
ref_i = sin(2*pi*fm2*R2.t); ref_q = cos(2*pi*fm2*R2.t);
tau   = 20e-3;                                  % lock-in time constant
fc    = 1/(2*pi*tau);
X  = lp1(lp1(2*R2.vpre_ac.*ref_i, fc, dt2), fc, dt2);
Y  = lp1(lp1(2*R2.vpre_ac.*ref_q, fc, dt2), fc, dt2);
phi = 2*atan(fm2/P.flp);                        % two-pole preamp lag
LI  = X*cos(phi) - Y*sin(phi);
Boff2 = (-Bspan + 2*Bspan*R2.t/Tsw);

% Expected absorption line (Lorentzian convolved with the field distribution)
Bg  = linspace(-Bspan, Bspan, 1201)';
D   = P.gamma*(Bg + P2.dB');                    % rad/s offsets
L   = 1./(1 + (D*P.T2).^2 + s);
line_exp = L*P2.wiso; line_exp = line_exp/max(line_exp);

f3 = figure('Name','Lock-in','Color','w','Position',[200 200 900 450]);
yyaxis left;  plot(Boff2*1e6, LI*1e3, 'LineWidth',1); ylabel('lock-in X (mV)');
yyaxis right; plot(Bg*1e6, line_exp, '--', 'LineWidth',1); ylabel('expected absorption (norm.)');
grid on; xlabel('B_0 offset from resonance (\muT)');
title(sprintf('Lock-in output: derivative of absorption (f_m = %d Hz, B_{mod} = %.1f \\muT, \\tau = %g ms)', ...
      fm2, bm2*1e6, tau*1e3));
exportgraphics(f3, fullfile(outdir,'fig3_lockin.png'), 'Resolution', 150);

% Lock-in summary
[~,imx] = max(LI); [~,imn] = min(LI);
pp = Boff2(imn) - Boff2(imx);
fprintf('  derivative peak-to-peak width: %.1f uT  (%.0f Hz)\n', abs(pp)*1e6, ...
        abs(pp)*P.gamma/(2*pi));
nz = std(LI(Boff2 < -0.7*Bspan));
fprintf('  lock-in SNR (pk-pk / noise rms): %.0f\n', (max(LI)-min(LI))/nz);
fprintf('\nFigures written to %s\n', outdir);

%% ================= Local functions =================
function P = set_isochromats(P, N)
    P.Niso = N;
    P.dB   = linspace(-4, 4, N)'*P.sigB;
    w      = exp(-0.5*(P.dB/P.sigB).^2);
    P.wiso = w/sum(w);
end

function R = run_sim(P, dt, Tend, Bext)
% Fixed-step RK4 on [Mx; My; Mz; A] (magnetisation normalised to M0 = 1)
    n = round(Tend/dt); N = P.Niso;
    y = [zeros(N,1); zeros(N,1); ones(N,1); P.A0];
    R.t = (0:n-1)'*dt; R.A = zeros(n,1); R.chi2 = zeros(n,1); R.Bmod = zeros(n,1);
    for k = 1:n
        t = R.t(k);
        [k1, c] = deriv(t,        y,            P, Bext);
        k2      = deriv(t+dt/2,   y+dt/2*k1,    P, Bext);
        k3      = deriv(t+dt/2,   y+dt/2*k2,    P, Bext);
        k4      = deriv(t+dt,     y+dt*k3,      P, Bext);
        R.A(k) = y(end); R.chi2(k) = c; R.Bmod(k) = Bext(t);
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    end
end

function [dy, chi2] = deriv(t, y, P, Bext)
    N  = P.Niso;
    Mx = y(1:N); My = y(N+1:2*N); Mz = y(2*N+1:3*N); A = max(y(end), 1e-9);
    D  = P.gamma*(P.Bres + Bext(t) + P.dB) - P.w0;   % rotating-frame offsets, rad/s
    w1 = P.gamma*P.kB1*A;
    dMx = D.*My - Mx/P.T2;
    dMy = -D.*Mx + w1*Mz - My/P.T2;
    dMz = -w1*My - (Mz - 1)/P.T1;
    Mv   = P.wiso'*My;                                % ensemble absorption component
    chi2 = P.chi0*P.Bres*Mv/(2*P.kB1*A);              % = mu0*Mv_phys/(2*B1)
    dA   = (P.w0/(2*P.Q0))*(P.eps - (A/P.As)^2)*A - (P.w0/2)*P.eta*chi2*A;
    dy   = [dMx; dMy; dMz; dA];
end

function R = detector_chain(R, P, dt)
% Ideal envelope detector + white noise, two-pole preamp low-pass, AC coupling.
    fs   = 1/dt;
    enbw = pi*P.flp/4;                          % two cascaded 1st-order poles
    sw   = P.noise*sqrt(fs/(2*enbw));
    vdet = R.A + sw*randn(size(R.A));
    v    = P.Gpre*lp1(lp1(vdet - P.A0, P.flp, dt), P.flp, dt);
    R.vpre_ac = v - lp1(v, 2, dt);              % 2 Hz AC coupling
    % for short runs the 2 Hz high-pass has not settled: use mean removal instead
    if R.t(end) < 1, R.vpre_ac = v - mean(v(round(end/2):end)); end
end

function y = lp1(x, fc, dt)
    a  = 1 - exp(-2*pi*fc*dt);
    y  = filter(a, [1, -(1-a)], x, (1-a)*x(1));
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
