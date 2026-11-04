%% magnet_design.m - field level and homogeneity of the NdFeB H-frame magnet
% Uses magnet_fd3d (3D finite-difference magnetostatics, 1/8 symmetry).
% 1) validates the solver against two exact cases,
% 2) sweeps the gap for the baseline design,
% 3) compares baseline / Rose-rim / large-pole designs,
% 4) writes magnet_design.png and magnet_design_results.mat.
% Run time: a few minutes (0.5 mm grid for the final comparison).

here = fileparts(mfilename('fullpath'));
gam  = 42.577;                       % MHz/T
ftun = [16.4 22.6];                  % oscillator tuning range, MHz

%% 1) validation
G  = struct('slab',true,'pad',0,'Dy',5,'Hy',5,'tb',0,'g',20,'tm',15,'tc',5,'tp',20);
q  = magnet_fd3d(G,1);
ex = 1.3*(15/1.05)/(10 + 15/1.05 + 25/1000);
fprintf('Validation, 1D slab circuit : FD %.4f T, exact %.4f T (%+.2f %%)\n', q.B0, ex, 100*(q.B0/ex-1));
disc = @(zz) 1.3/2*((zz+15)./sqrt(25^2+(zz+15).^2) - zz./sqrt(25^2+zz.^2));
q  = magnet_fd3d(struct('g',20,'yoke',false,'tc',0),1);
fprintf('Validation, bare Ø50 discs  : FD %.4f T, exact %.4f T (%+.2f %%)\n', q.B0, 2*disc(10), 100*(q.B0/(2*disc(10))-1));

%% 2) gap sweep, baseline (Ø50 x 15 mm N42, 5 mm caps, 20 mm yoke)
gaps = 14:2:34; B0g = zeros(size(gaps)); ppg = B0g;
for k = 1:numel(gaps)
    q = magnet_fd3d(struct('g',gaps(k)),1); B0g(k) = q.B0; ppg(k) = q.sample_ppm_pp;
end

%% 3) design comparison (0.5 mm grid)
D = {
 'Baseline: Ø50 poles, 20 mm gap',          struct('g',20)
 'A: + Rose rim 1.5 x 2 mm',                struct('g',20,'rimh',1.5,'rimw',2)
 'B: Ø80 poles, 30 mm yoke, 22 mm gap',     struct('g',22,'Rm',40,'Dy',50,'tp',30,'tb',30)
};
samples = {'Ø10 x 10 mm',[5 5]; 'Ø5 x 5 mm',[2.5 2.5]};
res = struct('name',{},'sample',{},'B0',{},'f0',{},'pp',{},'sigma_uT',{},'steel',{});
RD = cell(size(D,1),1);
for k = 1:size(D,1)
    for s = 1:size(samples,1)
        geo = D{k,2}; geo.rs = samples{s,2}(1); geo.zs = samples{s,2}(2);
        q = magnet_fd3d(geo,0.5);
        if s == 1, RD{k} = q; end
        res(end+1) = struct('name',D{k,1},'sample',samples{s,1},'B0',q.B0,'f0',gam*q.B0, ...
            'pp',q.sample_ppm_pp,'sigma_uT',q.sample_ppm_std*q.B0,'steel',q.Bsteel_p99); %#ok<SAGROW>
    end
end
T = struct2table(res);
disp(T)
save(fullfile(here,'magnet_design_results.mat'),'T','gaps','B0g','ppg','RD','D');

%% 4) figure  (to redraw only: load magnet_design_results.mat, then run this section)
f = figure('Color','w','Position',[100 100 1300 820],'Visible','off');
tl = tiledlayout(f,2,3,'TileSpacing','compact','Padding','compact');

% (a) field map in the x-z plane (y = 0), baseline vs design A, in ppm
for k = 1:2
    q = RD{k}; ax = nexttile(tl);
    xx = [-flipud(q.x); q.x]; zz = [-flipud(q.z); q.z];
    Bp = squeeze(q.Bx(:,1,:)); Bp = [flipud(Bp); Bp]; Bp = [fliplr(Bp), Bp];
    sel_x = abs(xx) <= q.geo.g/2; sel_z = abs(zz) <= 25;
    imagesc(ax, xx(sel_x), zz(sel_z), ((Bp(sel_x,sel_z)'/q.B0) - 1)*1e6);
    axis(ax,'xy','equal','tight'); clim(ax,[-3000 3000]); colormap(ax, turbo);
    hold(ax,'on'); rectangle(ax,'Position',[-5 -5 10 10],'EdgeColor','w','LineWidth',1.5);
    rectangle(ax,'Position',[-2.5 -2.5 5 5],'EdgeColor','w','LineStyle','--');
    xlabel(ax,'x (mm, along B_0)'); ylabel(ax,'z (mm, tube axis)');
    title(ax, {D{k,1}, sprintf('B_0 = %.3f T, (B/B_0 - 1) in ppm', q.B0)},'FontWeight','normal');
    cb = colorbar(ax); cb.Label.String = 'ppm';
end

% (b) profiles through the centre
ax = nexttile(tl); hold(ax,'on'); grid(ax,'on');
cols = lines(3);
for k = 1:3
    q = RD{k};
    plot(ax, [-flipud(q.x); q.x], ([flipud(q.Bx(:,1,1)); q.Bx(:,1,1)]/q.B0 - 1)*1e6, '-', 'Color',cols(k,:), 'LineWidth',1.5);
    plot(ax, [-flipud(q.z); q.z], ([flipud(squeeze(q.Bx(1,1,:))); squeeze(q.Bx(1,1,:))]/q.B0 - 1)*1e6, '--', 'Color',cols(k,:), 'LineWidth',1.5);
end
xlim(ax,[-8 8]); ylim(ax,[-3000 3000]); xline(ax,[-5 5],':'); xline(ax,[-2.5 2.5],':');
xlabel(ax,'distance from centre (mm)'); ylabel(ax,'(B/B_0 - 1) (ppm)');
legend(ax, {'baseline, along x','baseline, along z','A rim, x','A rim, z','B Ø80, x','B Ø80, z'},'Location','southoutside','NumColumns',3);
title(ax,'Field profiles (solid: along B_0, dashed: along tube)','FontWeight','normal');

% (c) gap sweep
ax = nexttile(tl,[1 3]);
yyaxis(ax,'left');
fill(ax,[gaps(1) gaps(end) gaps(end) gaps(1)], ftun([1 1 2 2])/gam, [0.85 0.95 0.85],'EdgeColor','none','FaceAlpha',0.6); hold(ax,'on');
plot(ax,gaps,B0g,'o-','LineWidth',1.5); ylabel(ax,'B_0 at centre (T)');
yline(ax,0.4697,'--','20.0 MHz');
yyaxis(ax,'right'); plot(ax,gaps,ppg,'s-','LineWidth',1.5); set(ax,'YScale','log'); ylabel(ax,'p-p over Ø10x10 mm sample (ppm)');
xlabel(ax,'gap between pole caps (mm)'); grid(ax,'on');
title(ax,'Baseline (Ø50 x 15 mm N42, 5 mm caps, no rim): B_0 and homogeneity vs gap. Green band: oscillator tuning range 16.4-22.6 MHz','FontWeight','normal');

exportgraphics(f, fullfile(here,'magnet_design.png'),'Resolution',150);
close(f);
fprintf('Wrote magnet_design.png\n');
