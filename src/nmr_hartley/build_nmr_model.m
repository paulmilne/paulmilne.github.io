function build_nmr_model()
% build_nmr_model - creates nmr_marginal_osc.slx, a system-level Simulink
% model of the JFET Hartley marginal-oscillator NMR spectrometer:
%   B0 sweep + modulation coils -> Bloch isochromat ensemble coupled to an
%   averaged marginal-oscillator amplitude -> detector/preamp -> scope YT,
%   scope XY and lock-in.
% Parameters live in nmr_params.m (run as the model InitFcn).

mdl  = 'nmr_marginal_osc';
here = fileparts(mfilename('fullpath'));
if bdIsLoaded(mdl), close_system(mdl, 0); end
f = fullfile(here, [mdl '.slx']);
if exist(f, 'file'), delete(f); end

evalin('base', 'nmr_params');
new_system(mdl);
set_param(mdl, 'Solver','ode45', 'StopTime','P.Tstop', 'MaxStep','P.maxStep', ...
    'RelTol','1e-5', 'PreLoadFcn','nmr_params', 'InitFcn','nmr_params', ...
    'ReturnWorkspaceOutputs','on');

b = @(lib, name, varargin) add_block(lib, [mdl '/' name], varargin{:});

% ---- Field: B0 offset/sweep and modulation coils ----
b('simulink/Sources/Ramp', 'B0 offset and sweep', 'slope','P.sweep', 'start','0', 'X0','P.Boff');
b('simulink/Sources/Sine Wave', 'Modulation field', 'Amplitude','P.Bmod', ...
  'Frequency','2*pi*P.fm', 'Phase','0');
b('simulink/Math Operations/Gain', 'Sense resistor', 'Gain','P.Rsense/P.kcoil');
b('simulink/Math Operations/Sum', 'Field at sample', 'Inputs','++');

% ---- Spins + oscillator ----
b('simulink/User-Defined Functions/MATLAB Function', 'Bloch ensemble and oscillator');
b('simulink/Continuous/Integrator', 'State integrator', 'InitialCondition','P.y0');

code = strjoin({
'function [dy, A, chi2] = spins_osc(y, Bext, P)'
'% Rotating-frame Bloch equations for an isochromat ensemble, coupled to'
'% the averaged amplitude A of a marginal oscillator.'
'N  = numel(P.dB);'
'Mx = y(1:N); My = y(N+1:2*N); Mz = y(2*N+1:3*N);'
'A  = max(y(end), 1e-9);'
'D  = P.gamma*(P.Bres + Bext + P.dB) - P.w0;   % offsets, rad/s'
'w1 = P.gamma*P.kB1*A;                         % RF nutation rate'
'dMx = D.*My - Mx/P.T2;'
'dMy = -D.*Mx + w1*Mz - My/P.T2;'
'dMz = -w1*My - (Mz - 1)/P.T1;'
'chi2 = P.chi0*P.Bres*(P.wiso''*My)/(2*P.kB1*A);  % absorption chi'''''
'dA = (P.w0/(2*P.Q0))*(P.eps - (A/P.As)^2)*A - (P.w0/2)*P.eta*chi2*A;'
'dy = [dMx; dMy; dMz; dA];'
}, newline);
rt = sfroot;
ch = rt.find('-isa','Stateflow.EMChart', 'Path',[mdl '/Bloch ensemble and oscillator']);
ch.Script = code;
d = ch.find('-isa','Stateflow.Data', 'Name','P');
d.Scope = 'Parameter'; d.Tunable = false;

% ---- Detector and preamp ----
b('simulink/Sources/Band-Limited White Noise', 'Detector noise', 'Cov','P.noisePower', ...
  'Ts','P.Tsn', 'seed','23341');
b('simulink/Sources/Constant', 'No-signal level A0', 'Value','P.A0');
b('simulink/Math Operations/Sum', 'Diode detector', 'Inputs','++-');
b('simulink/Math Operations/Gain', 'Preamp gain', 'Gain','P.Gpre');
b('simulink/Continuous/Transfer Fcn', 'Preamp LPF', 'Numerator','1', ...
  'Denominator','conv([1/(2*pi*P.flp) 1],[1/(2*pi*P.flp) 1])');
b('simulink/Continuous/Transfer Fcn', 'AC coupling', 'Numerator','[1 0]', ...
  'Denominator','[1 2*pi*P.fhp]');

% ---- Lock-in ----
b('simulink/Sources/Sine Wave', 'Reference', 'Amplitude','2', ...
  'Frequency','2*pi*P.fm', 'Phase','-P.phi');
b('simulink/Math Operations/Product', 'Mixer');
b('simulink/Continuous/Transfer Fcn', 'Lock-in LPF', 'Numerator','1', ...
  'Denominator','conv([P.tau 1],[P.tau 1])');

% ---- Instruments and logging ----
b('simulink/Sinks/Scope', 'Oscilloscope YT', 'NumInputPorts','2');
b('simulink/Sinks/XY Graph', 'Oscilloscope XY');
b('simulink/Sinks/Scope', 'Lock-in output');
logs = {'A','ch1','ch2','lockin','chi2','Bext'};
for k = 1:numel(logs)
    b('simulink/Sinks/To Workspace', ['log ' logs{k}], 'VariableName',logs{k}, ...
      'SaveFormat','Timeseries');
end

% ---- Wiring ----
L = @(s, dd) add_line(mdl, s, dd, 'autorouting','smart');
L('B0 offset and sweep/1', 'Field at sample/1');
L('Modulation field/1',    'Field at sample/2');
L('Modulation field/1',    'Sense resistor/1');
L('Field at sample/1',     'Bloch ensemble and oscillator/2');
L('State integrator/1',    'Bloch ensemble and oscillator/1');
L('Bloch ensemble and oscillator/1', 'State integrator/1');
L('Bloch ensemble and oscillator/2', 'Diode detector/1');
L('Detector noise/1',      'Diode detector/2');
L('No-signal level A0/1',  'Diode detector/3');
L('Diode detector/1',      'Preamp gain/1');
L('Preamp gain/1',         'Preamp LPF/1');
L('Preamp LPF/1',          'AC coupling/1');
L('Sense resistor/1',      'Oscilloscope YT/1');
L('AC coupling/1',         'Oscilloscope YT/2');
L('Sense resistor/1',      'Oscilloscope XY/1');
L('AC coupling/1',         'Oscilloscope XY/2');
L('AC coupling/1',         'Mixer/1');
L('Reference/1',           'Mixer/2');
L('Mixer/1',               'Lock-in LPF/1');
L('Lock-in LPF/1',         'Lock-in output/1');
L('Bloch ensemble and oscillator/2', 'log A/1');
L('Sense resistor/1',      'log ch1/1');
L('AC coupling/1',         'log ch2/1');
L('Lock-in LPF/1',         'log lockin/1');
L('Bloch ensemble and oscillator/3', 'log chi2/1');
L('Field at sample/1',     'log Bext/1');

% ---- Group into subsystems ----
groups = {
  'Modulation coils',     {'Modulation field','Sense resistor'}
  'Spins and oscillator', {'Bloch ensemble and oscillator','State integrator'}
  'Detector and preamp',  {'Detector noise','No-signal level A0','Diode detector', ...
                           'Preamp gain','Preamp LPF','AC coupling'}
  'Lock-in amplifier',    {'Reference','Mixer','Lock-in LPF'}
};
for g = 1:size(groups,1)
    h = cellfun(@(n) get_param([mdl '/' n], 'Handle'), groups{g,2});
    Simulink.BlockDiagram.createSubsystem(h);
    ss = find_system(mdl, 'SearchDepth',1, 'BlockType','SubSystem', 'Name','Subsystem');
    set_param(ss{1}, 'Name', groups{g,1});
end

Simulink.BlockDiagram.arrangeSystem(mdl);
for g = 1:size(groups,1)
    Simulink.BlockDiagram.arrangeSystem([mdl '/' groups{g,1}]);
end

save_system(mdl, f);
fprintf('Saved %s\n', f);
end
