function render_setup_3d()
% render_setup_3d - 3D illustration of the bench set-up (not a CAD model).
%   Permanent-magnet H-frame yoke, NdFeB discs + soft-iron pole caps,
%   modulation (Helmholtz) coils wound on the pole caps, sample tube with the
%   RF coil, RF-head box, driver/preamp board and waveform generator.
% Units: mm.  B0 is along +x, the sample tube is vertical (z), so the RF
% field of the solenoid (along z) is perpendicular to B0.
% Writes setup_3d_overview.png and setup_3d_gap.png next to this file.

here = fileparts(mfilename('fullpath'));
C = struct('steel',[0.42 0.45 0.50], 'mag',[0.72 0.72 0.76], 'cap',[0.55 0.57 0.60], ...
    'cu',[0.80 0.45 0.20], 'glass',[0.85 0.93 1.00], 'liq',[0.15 0.45 0.95], ...
    'al',[0.80 0.82 0.85], 'pcb',[0.05 0.45 0.20], 'ic',[0.10 0.10 0.10], ...
    'gen',[0.25 0.27 0.30], 'screen',[0.10 0.25 0.35], 'table',[0.93 0.90 0.84]);

% ---------------- geometry ----------------
g  = 20;  tm = 15; tc = 5; Rm = 25;           % gap, magnet & cap thickness, radius
xc = g/2;  xm = xc + tc;  xp = xm + tm;       % cap face, magnet face, plate face
tp = 20;  H = 70;  D = 30;  tb = 20;           % plate thickness, half-height, half-depth, bar thickness
hole = 8;                                      % half-size of sample hole in top bar
ztab = -H - tb - 40;                           % table height

f = figure('Color','w','Position',[50 50 1500 950],'Visible','off');
ax = axes(f); hold(ax,'on'); axis(ax,'equal','off');

% table
box(ax,[-330 300],[-130 130],[ztab-4 ztab],C.table,1);

% ---- magnet yoke (H-frame) on a 40 mm plinth ----
box(ax,[-xp-tp -xp],[-D D],[-H H],C.steel,1);           % side plates
box(ax,[ xp  xp+tp],[-D D],[-H H],C.steel,1);
box(ax,[-xp-tp xp+tp],[-D D],[-H-tb -H],C.steel,1);     % bottom bar
box(ax,[-xp-tp -hole],[-D D],[H H+tb],C.steel,1);       % top bar, with square hole
box(ax,[ hole xp+tp],[-D D],[H H+tb],C.steel,1);
box(ax,[-hole hole],[-D -hole],[H H+tb],C.steel,1);
box(ax,[-hole hole],[ hole D],[H H+tb],C.steel,1);
box(ax,[-xp-tp-10 xp+tp+10],[-D-10 D+10],[ztab -H-tb],[0.6 0.45 0.3],1);  % wooden plinth

% ---- NdFeB discs + pole caps ----
for s = [-1 1]
    cyl(ax,'x',s*[xm xp],[0 0],Rm,C.mag,1);
    cyl(ax,'x',s*[xc xm],[0 0],Rm,C.cap,1);
    % modulation coil wound on the pole cap (Helmholtz: spacing ~ radius)
    ring(ax,'x',s*[xc+0.5 xm+4],Rm+0.5,Rm+3.5,C.cu);
end

% ---- sample tube, liquid, RF coil ----
rt = 5;
cyl(ax,'z',[-14 165],[0 0],rt,C.glass,0.25);
cyl(ax,'z',[-13 12],[0 0],rt-0.7,C.liq,0.85);
th = linspace(0, 8*2*pi, 800);
plot3(ax,(rt+0.6)*cos(th),(rt+0.6)*sin(th),linspace(-5,5,numel(th)),'Color',C.cu,'LineWidth',2.5);
plot3(ax,[rt+0.6 rt+0.6],[0 0],[5 H+tb+15],'Color',C.cu,'LineWidth',1.5);     % leads up to RF head
plot3(ax,[rt+0.6 rt+2],[0 2],[-5 H+tb+15],'Color',C.cu,'LineWidth',1.5);

% ---- RF head (shielded oscillator box) sitting on the top bar ----
zr = H + tb;
box(ax,[-35 35],[-28 28],[zr+15 zr+55],C.al,0.55);
bnc(ax,[35 0 zr+35],'x');                          % v_det out
text(ax,0,-30,zr+70,'RF head (Hartley oscillator + detector)','HorizontalAlignment','center','FontSize',11);

% ---- driver + preamp board ----
bx = [120 250]; by = [-50 50]; bz = ztab + 12;
box(ax,bx,by,[bz-1.6 bz],C.pcb,1);
for k = 0:3, box(ax,[135+28*k 155+28*k],[-10 0],[bz bz+4],C.ic,1); end     % DIP ICs / TO-126
for k = 0:4, cyl(ax,'z',[bz bz+10],[140+24*k 25],4,[0.15 0.25 0.6],1); end % capacitors
for k = 0:4, cyl(ax,'x',[134+24*k 146+24*k],[-30 bz+2],1.5,[0.85 0.75 0.55],1); end % resistors
box(ax,[230 245],[-10 10],[bz bz+14],[0.25 0.25 0.25],1);                 % TO-126 transistors
for yy = [-35 35], cyl(ax,'z',[ztab bz-1.6],[bx(1)+5 yy],2,[0.7 0.7 0.7],1); end
text(ax,185,-85,bz+2,'Coil driver + preamp (\pm12 V)','HorizontalAlignment','center','FontSize',11);

% ---- waveform generator ----
gx = [-320 -170]; gy = [-60 60]; gz = [ztab ztab+80];
box(ax,gx,gy,gz,C.gen,1);
box(ax,[-310 -230],[-60.5 -60],[ztab+25 ztab+70],C.screen,1);               % screen
plot3(ax,linspace(-305,-235,200),-61*ones(1,200),ztab+47+15*sin(linspace(0,6*pi,200)),'Color',[1 0.85 0.2],'LineWidth',1.5);
for k = 0:2, cyl(ax,'y',[-66 -60],[-210 ztab+60-18*k],5,[0.75 0.75 0.75],1); end
bnc(ax,[-190 -60 ztab+20],'y'); bnc(ax,[-215 -60 ztab+20],'y');            % CH1, CH2
text(ax,-245,-62,ztab+92,'Waveform generator (2 ch)','HorizontalAlignment','center','FontSize',11);
text(ax,-215,-75,ztab+10,'CH2  CH1','FontSize',8);

% ---- cables ----
cable(ax,[-190 -68 ztab+20; -120 -120 ztab+5; 60 -100 ztab+5; 135 -40 bz+2],[0.1 0.1 0.1]);   % CH1 -> board
cable(ax,[-215 -68 ztab+20; -140 -125 ztab+3; 70 -110 ztab+3; 140 -45 bz+2],[0.6 0.1 0.1]);   % CH2 -> board
cable(ax,[240 0 bz+5; 200 60 bz+40; 90 50 0; xp+tp 20 10],[0.8 0.45 0.2]);                    % coil drive
cable(ax,[43 0 zr+35; 110 0 zr+30; 180 30 bz+60; 200 30 bz+2],[0.1 0.1 0.1]);                 % v_det -> preamp

% ---- labels in the scene ----
quiver3(ax,-8,-D-25,0,16,0,0,0,'k','LineWidth',2,'MaxHeadSize',2);
text(ax,0,-D-25,7,'B_0','FontSize',12,'FontWeight','bold','HorizontalAlignment','center');
text(ax,-xp-tp-8,-D,H,'steel yoke','HorizontalAlignment','right','FontSize',11);
text(ax,0,-95,ztab+2,'NdFeB H-frame magnet, 20 mm gap, \approx0.47 T','HorizontalAlignment','center','FontSize',11);

lights(ax);
view(ax,-38,22); camva(ax,9.6); camtarget(ax,[-20 0 -5]);
labels_to_front(ax);
exportgraphics(f, fullfile(here,'setup_3d_overview.png'),'Resolution',160);

% ---------------- close-up of the gap (front half of yoke removed) ----------------
delete(findobj(ax,'Type','text')); delete(findobj(ax,'Type','quiver'));
hs = findobj(ax,'Tag','yoke');  %#ok<NASGU>
set(findobj(ax,'Type','patch','-and','FaceColor',C.steel),'FaceAlpha',0.18,'EdgeAlpha',0.15);
quiver3(ax,-6,-22,-20,12,0,0,0,'k','LineWidth',2,'MaxHeadSize',2);
text(ax,0,-22,-14,'B_0','FontSize',13,'FontWeight','bold','HorizontalAlignment','center');
quiver3(ax,0,-14,14,0,0,10,0,'Color',[0.8 0.1 0.1],'LineWidth',2,'MaxHeadSize',2);
text(ax,-3,-14,30,'B_1 (RF coil axis)','FontSize',11,'Color',[0.8 0.1 0.1],'HorizontalAlignment','right');
text(ax, 32,-34,-34,'NdFeB disc Ø50 × 15 mm','FontSize',11);
text(ax, 18,-34, 36,'soft-iron pole cap + modulation coil','FontSize',11);
text(ax,-58,-34,-42,{'CuSO_4-doped water in Ø10 mm tube','8-turn RF coil (0.8 µH)'},'FontSize',11);
xlim(ax,[-60 60]); ylim(ax,[-45 45]); zlim(ax,[-48 48]);
camva(ax,'auto'); camtarget(ax,'auto'); view(ax,-20,12); camva(ax,6.5);
labels_to_front(ax);
exportgraphics(f, fullfile(here,'setup_3d_gap.png'),'Resolution',160);
close(f);
fprintf('Wrote setup_3d_overview.png and setup_3d_gap.png\n');
end

% ================= helpers =================
function labels_to_front(ax)
% Slide each text label along the line of sight towards the camera so it is
% drawn in front of all geometry; its on-screen position is unchanged.
    % axes use orthographic projection, so shift along the view direction
    d = campos(ax) - camtarget(ax);
    for t = findobj(ax,'Type','text')'
        t.Position = t.Position + 0.8*d;
        if ~isequal(t.Color,[1 1 1]), set(t,'BackgroundColor','w','Margin',1); end
    end
end

% ================= primitives =================
function box(ax,x,y,z,c,a)
    [X,Y,Z] = ndgrid(x,y,z);
    V = [X(:) Y(:) Z(:)];
    F = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
    patch(ax,'Vertices',V,'Faces',F,'FaceColor',c,'FaceAlpha',a,'EdgeColor',c*0.6,'EdgeAlpha',0.5);
end

function cyl(ax,axis_,span,ctr,r,c,a)
    % cylinder along axis_ from span(1) to span(2), centred at ctr (the two other coords)
    n = 48; th = linspace(0,2*pi,n);
    [T,S] = meshgrid(th,span);
    U = ctr(1) + r*cos(T); W = ctr(2) + r*sin(T);
    capU = ctr(1) + r*cos(th); capW = ctr(2) + r*sin(th);
    switch axis_
        case 'x', P = {S,U,W}; cp = @(s) {s*ones(1,n),capU,capW};
        case 'y', P = {U,S,W}; cp = @(s) {capU,s*ones(1,n),capW};
        case 'z', P = {U,W,S}; cp = @(s) {capU,capW,s*ones(1,n)};
    end
    surf(ax,P{:},'FaceColor',c,'EdgeColor','none','FaceAlpha',a);
    for s = span, q = cp(s); patch(ax,q{:},c,'EdgeColor','none','FaceAlpha',a); end
end

function ring(ax,axis_,span,ri,ro,c)
    % annular winding along axis_
    cyl(ax,axis_,span,[0 0],ro,c,1);
    n = 48; th = linspace(0,2*pi,n);
    [T,S] = meshgrid(th,span);
    switch axis_, case 'x'
        surf(ax,S,ri*cos(T),ri*sin(T),'FaceColor',c*0.8,'EdgeColor','none');
    end
end

function bnc(ax,p,axis_)
    switch axis_
        case 'x', cyl(ax,'x',[p(1) p(1)+8],p(2:3),4,[0.85 0.80 0.55],1);
        case 'y', cyl(ax,'y',[p(2)-8 p(2)],p([1 3]),4,[0.85 0.80 0.55],1);
    end
end

function cable(ax,P,c)
    % smooth cable through control points (Catmull-Rom style via pchip)
    t = 1:size(P,1); tt = linspace(1,size(P,1),200);
    plot3(ax,pchip(t,P(:,1),tt),pchip(t,P(:,2),tt),pchip(t,P(:,3),tt),'Color',c,'LineWidth',2.5);
end

function lights(ax)
    lighting(ax,'gouraud'); material(ax,'dull');
    light(ax,'Position',[-1 -1 1.5],'Style','infinite');
    light(ax,'Position',[1 -0.5 0.6],'Style','infinite','Color',[0.5 0.5 0.5]);
end
