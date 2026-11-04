% verify_nmr_model.m - runs nmr_marginal_osc.slx in each mode and checks the
% results against nmr_marginal_osc_sim.m (dip ~5.8 %, derivative p-p ~12 uT).
mdl  = 'nmr_marginal_osc';
here = fileparts(mfilename('fullpath'));
load_system(fullfile(here, [mdl '.slx']));

nmr_mode = 'scope';     tic; o1 = sim(mdl); t1 = toc;
nmr_mode = 'scope_off'; o2 = sim(mdl);
nmr_mode = 'lockin';    tic; o3 = sim(mdl); t3 = toc;
nmr_mode = 'scope';     nmr_params;          % leave the default mode loaded

A = o1.A;  ta = A.Time; a = squeeze(A.Data);
fprintf('Scope mode   : %.1f s, max dip %.2f %% of A0 (script: 5.79 %%)\n', ...
        t1, 100*(1 - min(a(ta >= 0.02))/0.5));
A2 = o2.A; a2 = squeeze(A2.Data);
fprintf('Offset mode  : max dip %.2f %% of A0 (script: 5.91 %%)\n', ...
        100*(1 - min(a2(A2.Time >= 0.02))/0.5));
li = o3.lockin; tl = li.Time; Lv = squeeze(li.Data);
Bsw = -30e-6 + (60e-6/1.5)*tl;
[~, i1] = max(Lv); [~, i2] = min(Lv);
fprintf('Lock-in mode : %.1f s, derivative p-p width %.1f uT (script: 12.0 uT)\n', ...
        t3, abs(Bsw(i1) - Bsw(i2))*1e6);

% Check plots from logged signals
fig = figure('Color','w','Position',[100 100 1000 700]);
tl2 = tiledlayout(fig, 2, 2, 'TileSpacing','compact');
for j = 1:2
    o = {o1, o2}; o = o{j};
    c1 = o.ch1; c2 = o.ch2; k1 = c1.Time >= 0.02; k2 = c2.Time >= 0.02;
    ax = nexttile(tl2);
    yyaxis(ax,'left');  plot(ax, (c1.Time(k1)-0.02)*1e3, squeeze(c1.Data(k1))*1e3); ylabel(ax,'CH1 (mV)');
    yyaxis(ax,'right'); plot(ax, (c2.Time(k2)-0.02)*1e3, squeeze(c2.Data(k2)));     ylabel(ax,'CH2 (V)');
    xlabel(ax,'time (ms)'); grid(ax,'on');
    title(ax, sprintf('Simulink YT, B_0 offset %+d \\muT', 20*(j-1)));
end
ax = nexttile(tl2);
c1 = o1.ch1; c2 = o1.ch2; k = c2.Time >= 0.02;
plot(ax, interp1(c1.Time, squeeze(c1.Data), c2.Time(k))*1e3, squeeze(c2.Data(k)));
xlabel(ax,'X = CH1 (mV)'); ylabel(ax,'Y = CH2 (V)'); grid(ax,'on'); title(ax,'Simulink XY, on resonance');
ax = nexttile(tl2);
plot(ax, Bsw*1e6, Lv*1e3); xlabel(ax,'B_0 offset (\muT)'); ylabel(ax,'lock-in (mV)');
grid(ax,'on'); title(ax,'Simulink lock-in sweep');
exportgraphics(fig, fullfile(here, 'simulink_check.png'), 'Resolution', 130);

% Diagram images
open_system(mdl);
saveas(get_param(mdl, 'Handle'), fullfile(here, 'model_top.png'));
saveas(get_param([mdl '/Spins and oscillator'], 'Handle'), fullfile(here, 'model_spins.png'));
